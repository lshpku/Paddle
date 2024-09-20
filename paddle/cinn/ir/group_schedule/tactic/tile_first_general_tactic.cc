// Copyright (c) 2024 CINN Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "paddle/cinn/ir/group_schedule/tactic/tile_first_general_tactic.h"
#include "paddle/cinn/adt/adt.h"
#include "paddle/cinn/common/integer_set.h"
#include "paddle/cinn/common/target.h"
#include "paddle/cinn/ir/ir.h"
#include "paddle/cinn/ir/ir_analyzer/ir_analyzer.h"
#include "paddle/cinn/ir/schedule/ir_schedule_util.h"

namespace cinn {
namespace ir {

using cinn::ir::analyzer::IsReductionSBlock;

class TileFirstGeneralTactic final : public ScheduleTactic {
 private:
  enum ApplyMethod {
    UNDEFINED,
    CONTINUOUS,
    DISCRETE,
    INTERVAL,
  };

 public:
  void Init(ScheduleContext* context) override;

  void Apply(ir::IRSchedule* sch, const std::string& block_id) override;

  std::string TacticName() const override { return "TileFirstGeneralTactic"; }

 private:
  void ApplyContinuousReduce(ir::IRSchedule* sch, const std::string& block_id);
  void ApplyDiscreteReduce(ir::IRSchedule* sch, const std::string& block_id);
  void ApplyIntervalReduce(ir::IRSchedule* sch, const std::string& block_id);

  void ReorderLoopsByLoopStrides(ir::IRSchedule* sch,
                                 const std::string& block_id);
  void FuseSpatialAxis(ir::IRSchedule* sch, const std::string& block_id);
  void FuseReduceAxis(ir::IRSchedule* sch, const std::string& block_id);
  void SplitSptialInner(ir::IRSchedule* sch, const std::string& block_id);
  void SplitReduceInner(ir::IRSchedule* sch, const std::string& block_id);
  void VariableTypeAssignment(ir::IRSchedule* sch, const std::string& block_id);
  void SetReduceType(ir::IRSchedule* sch, const std::string& block_id);
  void SetDiscreteReduceType(ir::IRSchedule* sch, const std::string& block_id);
  void SetIntervalReduceType(ir::IRSchedule* sch, const std::string& block_id);
  void BindCudaInfo(ir::IRSchedule* sch, const std::string& block_id);

 private:
  ScheduleContext* context_;
  ApplyMethod apply_method_;
  std::vector<int64_t> loop_permutation_;
  int64_t innermost_reduce_axis_count_;
  int64_t innermost_reduce_data_space_;
  std::unordered_map<std::string, std::string> map_rf_block_;
};

void TileFirstGeneralTactic::Init(ScheduleContext* context) {
  context_ = context;
  map_rf_block_.clear();
  auto& loop_strides = context_->config.base_info->loop_strides;
  auto& reduce_axis = context_->config.base_info->reduce_axis;

  // We use loop_permutation_ to reorder the loops. Loop with a larger stride
  // will be placed outer, and vice versa.
  loop_permutation_.clear();
  if (!loop_strides.empty()) {
    loop_permutation_.resize(loop_strides.size());
    std::iota(loop_permutation_.begin(), loop_permutation_.end(), 0);
    std::sort(loop_permutation_.begin(),
              loop_permutation_.end(),
              [&](int64_t a, int64_t b) {
                return loop_strides[a] > loop_strides[b];
              });
  }

  // Find the innermost reduce axes (must be continuous) and their data space.
  innermost_reduce_axis_count_ = 0;
  innermost_reduce_data_space_ = 1;
  for (int i = loop_permutation_.size() - 1; i >= 0; i--) {
    int64_t axis = loop_permutation_[i];
    if (loop_strides[axis] == 0) {
      continue;
    }
    if (reduce_axis.find(axis) == reduce_axis.end()) {
      break;
    }
    ++innermost_reduce_axis_count_;

    int64_t data_space = context_->config.base_info->data_space[axis];
    if (data_space == -1) {
      innermost_reduce_data_space_ = -1;
    } else if (innermost_reduce_data_space_ != -1) {
      innermost_reduce_data_space_ = std::max(data_space * loop_strides[axis],
                                              innermost_reduce_data_space_);
    }
  }

  if (innermost_reduce_axis_count_ > 0) {
    if (innermost_reduce_data_space_ <= 16 &&
        innermost_reduce_data_space_ != -1) {
      apply_method_ = INTERVAL;
    } else {
      apply_method_ = CONTINUOUS;
    }
  } else {
    if (context_->config.base_info->reduce_axis.empty()) {
      apply_method_ = CONTINUOUS;
    } else {
      apply_method_ = DISCRETE;
    }
  }

  VLOG(4) << "After TileFirstGeneralTactic Init:\n"
          << "loop_permutation: " << utils::Join(loop_permutation_, ", ")
          << "innermost_reduce_axis_count: " << innermost_reduce_axis_count_
          << "innermost_reduce_data_space: " << innermost_reduce_data_space_
          << "apply_method: " << apply_method_;
}

void TileFirstGeneralTactic::Apply(ir::IRSchedule* sch,
                                   const std::string& block_id) {
  if (ir::IsReduceInitTensorName(block_id)) return;

  ReorderLoopsByLoopStrides(sch, block_id);
  VLOG(4) << "After ReorderLoopsByLoopStrides on block: [" << block_id
          << "], loop nest:\n"
          << sch->GetLoops(block_id)[0];

  switch (apply_method_) {
    case CONTINUOUS:
      ApplyContinuousReduce(sch, block_id);
      break;
    case DISCRETE:
      ApplyDiscreteReduce(sch, block_id);
      break;
    case INTERVAL:
      ApplyIntervalReduce(sch, block_id);
      break;
    default:
      PADDLE_THROW("The apply_method_ is invalid.")
  }
}

void TileFirstGeneralTactic::ApplyContinuousReduce(
    ir::IRSchedule* sch, const std::string& block_id) {
  const auto sp_thread = context_->config.tile_config.warp_num * 32 /
                         context_->config.tile_config.tree_reduce_num;
  const auto sp_loop = context_->config.tile_config.spatial_inner_num;
  const auto rd_thread = context_->config.tile_config.tree_reduce_num;
  VLOG(4) << "ApplyContinuousDataTile sp_thread=" << sp_thread;
  VLOG(4) << "ApplyContinuousDataTile sp_loop=" << sp_loop;
  VLOG(4) << "ApplyContinuousDataTile rd_thread=" << rd_thread;
  VLOG(4) << "ApplyContinuousDataTile vec_flatten_axis: "
          << utils::Join(vec_flatten_axis_, ", ");
  VLOG(4) << "ApplyContinuousDataTile vec_reduce_axis: "
          << utils::Join(vec_reduce_axis_, ", ");

  FuseReduceAxis(sch, block_id);
  VLOG(4) << "After FuseReduceAxis on block: [" << block_id << "], loop nest:\n"
          << sch->GetLoops(block_id)[0];

  FuseSpatialAxis(sch, block_id);
  VLOG(4) << "After FuseSpatialAxis on block: [" << block_id
          << "], loop nest:\n"
          << sch->GetLoops(block_id)[0];

  // Split spatial axes -> [sp_block, sp_loop, sp_thread]
  int current_reduce_axis = 0;
  if (vec_flatten_axis_.size() > 0) {
    auto loops = sch->GetLoops(block_id);
    if (sp_loop > 1 && sp_thread > 1) {
      // [S, R] => [S(-1), S(inner_loop), S(thread), R]
      sch->Split(loops[0], {-1, sp_loop, sp_thread});
      current_reduce_axis = 3;
    } else if (sp_loop > 1 || sp_thread > 1) {
      // [S, R] => [S(-1), S(thread), R]
      sch->Split(loops[0], {-1, sp_loop > 1 ? sp_loop : sp_thread});
      current_reduce_axis = 2;
    } else {
      // [S, R] => [S, R]
      current_reduce_axis = 1;
    }
  }
  VLOG(4) << "After SplitSptial on block: [" << block_id << "], loop nest:\n"
          << sch->GetLoops(block_id)[0];

  // Split reduce axes -> [rd_loop, rd_thread]
  if (vec_reduce_axis_.size() > 0) {
    auto loops = sch->GetLoops(block_id);
    // [S..S, R] => [S..S, R(-1), R(thread)]
    sch->Split(loops[current_reduce_axis], {-1, rd_thread});

    loops = sch->GetLoops(block_id);
    // [S..S, R(-1), R(thread)] => [S..S, R(thread), R(-1)]
    sch->Reorder({loops[current_reduce_axis + 1], loops[current_reduce_axis]});

    if (IsReductionSBlock(sch->GetBlock(block_id))) {
      loops = sch->GetLoops(block_id);
      ir::Expr rf_tensor =
          sch->FactorizeReduction(loops[current_reduce_axis],
                                  /* rf_axis = */ 0,
                                  /* with_write_back_block_init = */ false);
      map_rf_block_[block_id] = rf_tensor.as_tensor_ref()->name;
    }
  }
  VLOG(4) << "After SplitReduce on block: [" << block_id << "], loop nest:\n"
          << sch->GetLoops(block_id)[0];

  // Bind CUDA info
  const auto DoBind = [&](const std::vector<ir::Expr>& loops) {
    std::string sp_axis_type = "threadIdx.y";
    std::string rd_axis_type = "threadIdx.x";
    sch->Bind(loops[0], "blockIdx.x");
    if (!vec_flatten_axis_.empty() && sp_thread > 1) {
      if (vec_reduce_axis_.empty()) {
        // [S..S] => [S(blockIdx.x), optional(inner_loop), S(threadIdx.x)]
        sch->Bind(loops[current_reduce_axis - 1], rd_axis_type);
      } else {
        // [S..S, R..R] =>
        // [S(blockIdx.x), optional(inner_loop), S(threadIdx.y), R..R]
        sch->Bind(loops[current_reduce_axis - 1], sp_axis_type);
      }
    }
    if (!vec_reduce_axis_.empty() && current_reduce_axis > 0) {
      // [S(blockIdx.x), optional(inner_loop), S(threadIdx.y), R..R] =>
      // [S(blockIdx.x), optional(inner_loop), S(threadIdx.y), R(threadIdx.x),
      // R(inner_loop)]
      sch->Bind(loops[current_reduce_axis], rd_axis_type);
    }
  };
  DoBind(sch->GetLoops(block_id));
  if (map_rf_block_.count(block_id) > 0) {
    DoBind(sch->GetLoops(map_rf_block_[block_id]));
  }
  VLOG(4) << "After BindCudaInfo on block: [" << block_id << "], loop nest:\n"
          << sch->GetLoops(block_id)[0];

  VariableTypeAssignment(sch, block_id);
  SetReduceType(sch, block_id);
}

void TileFirstGeneralTactic::ApplyDiscreteReduce(ir::IRSchedule* sch,
                                                 const std::string& block_id) {
  FuseReduceAxis(sch, block_id);
  VLOG(4) << "After FuseReduceAxis on block: [" << block_id << "], loop nest:\n"
          << sch->GetLoops(block_id)[0];
  MergeDiscreteFlattenAxis(sch, block_id);
  VLOG(4) << "After MergeDiscreteFlattenAxis on block: [" << block_id
          << "], loop nest:\n"
          << sch->GetLoops(block_id)[0];
  SplitSptialInner(sch, block_id);
  VLOG(4) << "After SplitSptialInner on block: [" << block_id
          << "], loop nest:\n"
          << sch->GetLoops(block_id)[0];
  SplitReduceInner(sch, block_id);
  VLOG(4) << "After SplitReduceInner on block: [" << block_id
          << "], loop nest:\n"
          << sch->GetLoops(block_id)[0];
  BindCudaInfo(sch, block_id);
  VLOG(4) << "After BindCudaInfo on block: [" << block_id << "], loop nest:\n"
          << sch->GetLoops(block_id)[0];
  VariableTypeAssignment(sch, block_id);
  VLOG(4) << "After VariableTypeAssignment on block: [" << block_id
          << "], loop nest:\n"
          << sch->GetLoops(block_id)[0];
  SetDiscreteReduceType(sch, block_id);
}

void TileFirstGeneralTactic::ApplyIntervalReduce(ir::IRSchedule* sch,
                                                 const std::string& block_id) {
  const int64_t sp_thread = context_->config.tile_config.warp_num * 32 /
                            context_->config.tile_config.tree_reduce_num;
  const int64_t sp_loop = context_->config.tile_config.spatial_inner_num;
  const int64_t rd_thread = context_->config.tile_config.tree_reduce_num;
  const int64_t tx = innermost_reduce_data_space_;
  const int64_t ty = std::max(32 / tx, sp_thread);
  const int64_t tz = sp_thread * rd_thread / (tx * ty);
  VLOG(4) << "ApplyIntervalReduce tx=" << tx << " ty=" << ty << " tz=" << tz;

  FuseReduceAxis(sch, block_id);
  VLOG(4) << "After FuseReduceAxis on block: [" << block_id << "], loop nest:\n"
          << sch->GetLoops(block_id)[0];

  FuseSpatialAxis(sch, block_id);
  VLOG(4) << "After FuseSpatialAxis on block: [" << block_id
          << "], loop nest:\n"
          << sch->GetLoops(block_id)[0];

  // Split spatial axes -> [sp_block, sp_loop, ty]
  int current_reduce_axis = 0;
  if (vec_flatten_axis_.size() > 0) {
    auto loops = sch->GetLoops(block_id);
    if (sp_loop > 1 && ty > 1) {
      sch->Split(loops[0], {-1, sp_loop, ty});
      current_reduce_axis = 3;
    } else if (sp_loop > 1 || ty > 1) {
      sch->Split(loops[0], {-1, sp_loop > 1 ? sp_loop : ty});
      current_reduce_axis = 2;
    } else {
      current_reduce_axis = 1;
    }
  }
  VLOG(4) << "After SplitSptial on block: [" << block_id << "], loop nest:\n"
          << sch->GetLoops(block_id)[0];

  // Split reduce axes -> [rd_loop, tz, tx]
  {
    auto loops = sch->GetLoops(block_id);
    auto reduce_loop = loops[current_reduce_axis].As<ir::For>();
    sch->Split(loops[current_reduce_axis], {-1, tz * tx});

    loops = sch->GetLoops(block_id);
    sch->Reorder({loops[current_reduce_axis + 1], loops[current_reduce_axis]});

    if (IsReductionSBlock(sch->GetBlock(block_id))) {
      loops = sch->GetLoops(block_id);
      ir::Expr rf_tensor =
          sch->FactorizeReduction(loops[current_reduce_axis],
                                  /* rf_axis = */ 0,
                                  /* with_write_back_block_init = */ false);

      std::string rf_block_id = rf_tensor.as_tensor_ref()->name;
      map_rf_block_[block_id] = rf_block_id;
      loops = sch->GetLoops(rf_block_id);
      sch->Split(loops[current_reduce_axis], {tz, tx});
    }

    loops = sch->GetLoops(block_id);
    sch->Split(loops[current_reduce_axis], {tz, tx});
  }
  VLOG(4) << "After SplitReduce on block: [" << block_id << "], loop nest:\n"
          << sch->GetLoops(block_id)[0];

  // Bind CUDA info
  const auto DoBind = [&](const std::vector<ir::Expr>& loops) {
    sch->Bind(loops[0], "blockIdx.x");
    if (!vec_flatten_axis_.empty() && ty > 1) {
      sch->Bind(loops[current_reduce_axis - 1], "threadIdx.y");
    }
    sch->Bind(loops[current_reduce_axis], "threadIdx.z");
    sch->Bind(loops[current_reduce_axis + 1], "threadIdx.x");
  };
  DoBind(sch->GetLoops(block_id));
  if (map_rf_block_.count(block_id) > 0) {
    DoBind(sch->GetLoops(map_rf_block_[block_id]));
  }
  VLOG(4) << "After BindCudaInfo on block: [" << block_id << "], loop nest:\n"
          << sch->GetLoops(block_id)[0];

  VariableTypeAssignment(sch, block_id);
  SetIntervalReduceType(sch, block_id);
}

void TileFirstGeneralTactic::ReorderLoopsByLoopStrides(
    ir::IRSchedule* sch, const std::string& block_id) {
  if (loop_permutation_.empty()) {
    return;
  }
  auto& loop_strides = context_->config.base_info->loop_strides;
  auto& reduce_axis = context_->config.base_info->reduce_axis;

  // Reorder spatial and reduce loops seperately.
  std::vector<ir::Expr> loops = sch->GetLoops(block_id);
  std::vector<ir::Expr> sp_loops, rd_loops;
  for (auto axis : loop_permutation_) {
    if (reduce_axis.find(axis) != reduce_axis.end()) {
      rd_loops.push_back(loops[axis]);
    } else if (loop_strides[axis] != 0) {
      sp_loops.push_back(loops[axis]);
    }
  }
  sch->Reorder(sp_loops);
  sch->Reorder(rd_loops);
}

void TileFirstGeneralTactic::FuseSpatialAxis(ir::IRSchedule* sch,
                                             const std::string& block_id) {
  if (vec_flatten_axis_.size() >= 2) {
    sch->Fuse(block_id, vec_flatten_axis_);
  }
}

void TileFirstGeneralTactic::FuseReduceAxis(ir::IRSchedule* sch,
                                            const std::string& block_id) {
  std::vector<ir::Expr> loops = sch->GetLoops(block_id);
  int32_t max_loop_idx = 0;
  for (int32_t idx : vec_reduce_axis_) {
    max_loop_idx = std::max(max_loop_idx, idx);
    PADDLE_ENFORCE_EQ(idx < loops.size() || loops.size() == 1,
                      true,
                      ::common::errors::InvalidArgument(
                          "The reduce axis should meet: axis's idx < "
                          "loops.size() or loops.size() == 1, but received "
                          "idx= %d ,loops.size() = %d",
                          idx,
                          loops.size()));
  }
  if (max_loop_idx < loops.size() && vec_reduce_axis_.size() >= 2) {
    sch->Fuse(block_id, vec_reduce_axis_);
  }
}

void TileFirstGeneralTactic::SplitSptialInner(ir::IRSchedule* sch,
                                              const std::string& block_id) {
  auto loops = sch->GetLoops(block_id);
  if (loops.size() == 3) {
    // [S, S', R] => [S, S'(-1), S'(32), R]
    auto split_loops = sch->Split(loops[1], std::vector<int>({-1, 32}));
    // [S, S'(-1), S'(32), R] => [S, S'(32), R]
    sch->Fuse(block_id, std::vector<int>{0, 1});
  } else if (loops.size() == 2) {
    // [S, R] => [S(-1), S(32), R]
    auto split_loops = sch->Split(loops[0], std::vector<int>({-1, 32}));
  }
}

void TileFirstGeneralTactic::SplitReduceInner(ir::IRSchedule* sch,
                                              const std::string& block_id) {
  auto loops = sch->GetLoops(block_id);
  // [S(-1), S(32), R] => [S(-1), S(32), R(16), R(-1)]
  sch->Split(loops[2], std::vector<int>{16, -1});

  loops = sch->GetLoops(block_id);
  if (IsReductionSBlock(sch->GetBlock(block_id))) {
    ir::Expr rf_tensor =
        sch->FactorizeReduction(loops[2],
                                0,
                                /* with_write_back_block_init = */ false);
    map_rf_block_[block_id] = rf_tensor.as_tensor_ref()->name;
  }
}

void TileFirstGeneralTactic::VariableTypeAssignment(
    ir::IRSchedule* sch, const std::string& block_id) {
  const auto IsOutputTensor = [&](const std::string& tensor_name) -> bool {
    return context_->output_names.count(tensor_name) > 0;
  };
  const auto HasConsumers = [&](const ir::Expr& block) -> bool {
    return !ir::analyzer::GetConsumerSBlocks(block, sch->GetRootBlock(block))
                .empty();
  };

  auto block = sch->GetBlock(block_id);
  if (!IsOutputTensor(block_id) && HasConsumers(block)) {
    sch->SetBuffer(block, "local", false);
  }

  if (map_rf_block_.count(block_id) > 0) {
    auto block = sch->GetBlock(map_rf_block_[block_id]);
    sch->SetBuffer(block, "local", false);
  }
}

void TileFirstGeneralTactic::SetReduceType(ir::IRSchedule* sch,
                                           const std::string& block_id) {
  if (IsReductionSBlock(sch->GetBlock(block_id))) {
    auto block = sch->GetBlock(block_id)
                     .As<ir::ScheduleBlockRealize>()
                     ->schedule_block.As<ir::ScheduleBlock>();
    block->reduce_method = context_->config.tile_config.reduce_method;
  }
}

void TileFirstGeneralTactic::SetDiscreteReduceType(
    ir::IRSchedule* sch, const std::string& block_id) {
  if (IsReductionSBlock(sch->GetBlock(block_id))) {
    auto block = sch->GetBlock(block_id)
                     .As<ir::ScheduleBlockRealize>()
                     ->schedule_block.As<ir::ScheduleBlock>();
    block->reduce_method = cinn::ir::DiscreteReduceMethod();
  }
}

void TileFirstGeneralTactic::SetIntervalReduceType(
    ir::IRSchedule* sch, const std::string& block_id) {
  if (IsReductionSBlock(sch->GetBlock(block_id))) {
    auto block = sch->GetBlock(block_id)
                     .As<ir::ScheduleBlockRealize>()
                     ->schedule_block.As<ir::ScheduleBlock>();
    block->reduce_method = cinn::ir::IntervalReduceMethod();
  }
}

void TileFirstGeneralTactic::BindCudaInfo(ir::IRSchedule* sch,
                                          const std::string& block_id) {
  auto loops = sch->GetLoops(block_id);

  // [S(-1), S(32), R(16), R(-1)] =>
  // [S(blockIdx.x), S(threadIdx.x), R(threadIdx.y), R(inner_loop)]
  const auto DoBind = [&](const std::vector<ir::Expr>& loops) {
    sch->Bind(loops[0], "blockIdx.x");
    sch->Bind(loops[1], "threadIdx.x");
    sch->Bind(loops[2], "threadIdx.y");
  };

  DoBind(sch->GetLoops(block_id));

  if (map_rf_block_.count(block_id) > 0) {
    DoBind(sch->GetLoops(map_rf_block_[block_id]));
  }
}

std::unique_ptr<ScheduleTactic> CreateTileFirstGeneralTactic() {
  return std::make_unique<TileFirstGeneralTactic>();
}

}  // namespace ir
}  // namespace cinn

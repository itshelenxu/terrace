cc_library(
    name = "util",
    hdrs = ["internal/util.h"],
)

cc_library(
    name = "BitArray",
    hdrs = ["internal/BitArray.hpp"],
    deps = [
        "@ParallelTools//ParallelTools:parallel",
    ],
)

cc_library(
  name = "PMA_Lock",
  hdrs = ["PMA_Lock.hpp"],
  deps = [
    "util"
  ],
)

cc_library(
  name = "PMA",
  hdrs = ["PMA.hpp"],
  deps = [
    "PMA_Lock",
    "@ParallelTools//ParallelTools:parallel",
    "@parlaylib//parlay:primitives",
  ],
)

cc_library(
    name = "btree",
    hdrs = ["btree.h"],
)

cc_library(
  name = "terrace_graph",
  hdrs = ["terrace_graph.h"],
  deps = [
    "PMA",
    "btree",
    "@ParallelTools//ParallelTools:parallel",
  ]
)

package(default_visibility = ["//visibility:public"])

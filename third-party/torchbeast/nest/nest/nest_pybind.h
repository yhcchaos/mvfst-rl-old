/*
* Copyright (c) Facebook, Inc. and its affiliates.
* All rights reserved.
*
* This source code is licensed under the license found in the
* LICENSE file in the root directory of this source tree.
*
*/
#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "nest.h"

namespace pybind11 {
namespace detail {
template <typename Value>
struct type_caster<nest::Nest<Value>> {
  using ValueNest = nest::Nest<Value>;
  using value_conv = make_caster<Value>;

 public:
  PYBIND11_TYPE_CASTER(ValueNest, _("Nest[") + value_conv::name + _("]"));

  bool load(handle src, bool convert) {
    if (!src.ptr()) {
      return false;
    }
    if (isinstance<tuple>(src) || isinstance<list>(src)) {
      value.value = std::move(src).cast<std::vector<ValueNest>>();
      return true;
    }
    if (isinstance<dict>(src)) {
      value.value = std::move(src).cast<std::map<std::string, ValueNest>>();
      return true;
    }

    value_conv conv;
    if (!conv.load(src, convert)) return false;

    value.value = cast_op<Value&&>(std::move(conv));
    return true;
  }

  static handle cast(ValueNest&& src, return_value_policy policy,
                     handle parent) {
    return std::visit(
        nest::overloaded{
            [&policy, &parent](Value&& t) {
              return value_conv::cast(std::move(t), policy, parent);
            },
            [&policy, &parent](std::vector<ValueNest>&& v) {
              object py_list = reinterpret_steal<object>(
                  list_caster<std::vector<ValueNest>, ValueNest>::cast(
                      std::move(v), policy, parent));

              return handle(PyList_AsTuple(py_list.ptr()));
            },
            [&policy, &parent](std::map<std::string, ValueNest>&& m) {
              return map_caster<ValueNest, std::string, ValueNest>::cast(
                  std::move(m), policy, parent);
            }},
        std::move(src.value));
  }

  static handle cast(const ValueNest& src, return_value_policy policy,
                     handle parent) {
    return std::visit(
        nest::overloaded{
            [&policy, &parent](const Value& t) {
              return value_conv::cast(t, policy, parent);
            },
            [&policy, &parent](const std::vector<ValueNest>& v) {
              object py_list = reinterpret_steal<object>(
                  list_caster<std::vector<ValueNest>, ValueNest>::cast(
                      v, policy, parent));

              return handle(PyList_AsTuple(py_list.ptr()));
            },
            [&policy, &parent](const std::map<std::string, ValueNest>& m) {
              return map_caster<ValueNest, std::string, ValueNest>::cast(
                  m, policy, parent);
            }},
        src.value);
  }
};
}  // namespace detail
}  // namespace pybind11

#pragma once
#include <map>
#include <unordered_map>

using std::map;
using std::unordered_map;




template <class T>
struct IsMapLike {
  static constexpr bool value = false;
};

template <class K, class V>
struct IsMapLike<map<K, V>> {
  static constexpr bool value = true;
};

template <class K, class V>
struct IsMapLike<unordered_map<K, V>> {
  static constexpr bool value = true;
};

template <class T>
constexpr bool isMapLike(T&& x) {
  return IsMapLike<typename T>::value;
}

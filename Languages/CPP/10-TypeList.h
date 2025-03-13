#pragma once

#include <algorithm>
#include <cstddef>
#include <type_traits>
#include <utility>

template <typename... Types>
struct Typelist {};

template <typename... Ts>
constexpr size_t size(Typelist<Ts...>) {
  return sizeof...(Ts);
}

template <typename... Ts, typename... Us>
constexpr bool operator==(Typelist<Ts...>, Typelist<Us...>) {
  return false;
}

template <typename... Ts>
constexpr bool operator==(Typelist<Ts...>, Typelist<Ts...>) {
  return true;
}

template <typename... Ts, typename... Us>
constexpr bool operator!=(Typelist<Ts...> a, Typelist<Us...> b) {
  return !(a == b);
}

template <typename T, typename... Ts>
constexpr Typelist<Ts...> pop_front(Typelist<T, Ts...>) {
  return {};
}

template <typename T, typename... Ts>
constexpr Typelist<T, Ts...> push_front(Typelist<Ts...>) {
  return {};
}

template <typename T, typename... Ts>
constexpr Typelist<Ts..., T> push_back(Typelist<Ts...>) {
  return {};
}

template <template <typename> typename F, typename... Ts>
constexpr size_t find_if(Typelist<Ts...>) {
  bool pred[] = {F<Ts>::value...};
  return static_cast<size_t>(std::find(pred, pred + sizeof...(Ts), true) - pred);
}

template <typename T>
struct TypeEqual {
  template <typename U>
  struct Pred {
    static const bool value = std::is_same_v<T, U>;
  };
};

template <typename T, typename... Ts>
constexpr size_t find(Typelist<Ts...> typelist) {
  return find_if<TypeEqual<T>::template Pred>(typelist);
}

template <typename T, typename... Ts>
constexpr bool contains(Typelist<Ts...> typelist) {
  return find<T>(typelist) < sizeof...(Ts);
}

template <typename T>
struct get_impl;

template <size_t... Indices>
struct get_impl<std::index_sequence<Indices...>> {
  template <typename T>
  static constexpr T dummy(decltype(Indices, std::declval<void*>())..., T*, ...);
};

template <size_t Index, typename... Ts>
constexpr auto at(Typelist<Ts...>)
  requires(Index < sizeof...(Ts))
{
  static_assert(Index < sizeof...(Ts));
  return std::type_identity<decltype(get_impl<std::make_index_sequence<Index>>::dummy(
      static_cast<Ts*>(nullptr)...))>();
}

template <size_t N, typename T>
struct TypeCloner {
  using type = T;
};

template <typename T, typename U>
struct GenerateHelper;

template <size_t... Indices, typename U>
struct GenerateHelper<std::index_sequence<Indices...>, U> {
  using type = Typelist<typename TypeCloner<Indices, U>::type...>;
};

template <size_t N, typename T>
constexpr auto generate() {
  return typename GenerateHelper<std::make_index_sequence<N>, T>::type();
}

template <typename... Ts, typename... Us>
constexpr Typelist<Ts..., Us...> concat(Typelist<Ts...>, Typelist<Us...>) {
  return {};
}

template <typename... Ts>
struct IntermediateTypelist : Typelist<Ts...> {
  constexpr Typelist<Ts...> getTypelist() { return {}; }
};

template <typename... Ts, typename... Us>
constexpr IntermediateTypelist<Us..., Ts...> operator+(IntermediateTypelist<Ts...>,
                                                       IntermediateTypelist<Us...>) {
  return {};
}

template <typename... Ts>
constexpr auto reverse(Typelist<Ts...>) {
  return (IntermediateTypelist<>() + ... + IntermediateTypelist<Ts>()).getTypelist();
}

template <typename... Ts>
constexpr auto pop_back(Typelist<Ts...> typelist) {
  return reverse(pop_front(reverse(typelist)));
}

template <template <typename> typename F, typename... Ts>
constexpr Typelist<typename F<Ts>::type...> transform(Typelist<Ts...>) {
  return {};
}

template <typename... Ts, typename... Us>
constexpr IntermediateTypelist<Ts..., Us...> operator|(IntermediateTypelist<Ts...>,
                                                       IntermediateTypelist<Us...>) {
  return {};
}

template <template <typename> typename F, typename... Ts>
constexpr auto filter(Typelist<Ts...>) {
  return (IntermediateTypelist<>() | ... |
          std::conditional_t<F<Ts>::value, IntermediateTypelist<Ts>, IntermediateTypelist<>>())
      .getTypelist();
}

template <typename T, typename U, typename... Types>
struct UniqueHelper {
  using type = std::conditional_t<std::is_same_v<T, U>, IntermediateTypelist<U, Types...>,
                                  IntermediateTypelist<T, U, Types...>>;
};

template <typename T, typename U, typename... Types>
constexpr auto operator&(IntermediateTypelist<U, Types...>, IntermediateTypelist<T>) {
  return typename UniqueHelper<T, U, Types...>::type();
}

template <typename T>
constexpr IntermediateTypelist<T> operator&(IntermediateTypelist<>, IntermediateTypelist<T>) {
  return {};
}

template <typename... Ts>
constexpr auto unique(Typelist<Ts...>) {
  return reverse((IntermediateTypelist<>() & ... & IntermediateTypelist<Ts>()).getTypelist());
}

template <template <typename, typename> typename Comp, typename... Ts, typename... Us>
constexpr size_t compareElementwise(Typelist<Ts...>, Typelist<Us...>) {
  static_assert(sizeof...(Ts) == sizeof...(Us));
  bool res[] = {Comp<Ts, Us>::value...};
  size_t n = sizeof...(Ts);
  size_t ans = 0;
  for (size_t i = 0; i < n; ++i) {
    if (res[i]) ++ans;
  }
  return ans;
}

template <template <typename, typename> typename Comp, typename... Ts>
constexpr bool is_sorted(Typelist<Ts...> list) {
  constexpr size_t n = sizeof...(Ts);
  if constexpr (n <= 1) {
    return true;
  } else {
    return compareElementwise<Comp>(
               pop_front(push_back<typename decltype(at<0>(list))::type>(list)), list) ==
           Comp<typename decltype(at<0>(list))::type,
                typename decltype(at<n - 1>(list))::type>::value;
  }
}

template <template <typename, typename> typename Comp>
constexpr Typelist<> merge(Typelist<>, Typelist<>) {
  return {};
}

template <template <typename, typename> typename Comp, typename... Types>
constexpr Typelist<Types...> merge(Typelist<Types...>, Typelist<>) {
  return {};
}

template <template <typename, typename> typename Comp, typename... Types>
constexpr Typelist<Types...> merge(Typelist<>, Typelist<Types...>) {
  return {};
}

template <template <typename, typename> typename Comp, typename T, typename... Ts, typename U,
          typename... Us>
constexpr auto merge(Typelist<T, Ts...>, Typelist<U, Us...>) {
  if constexpr (Comp<U, T>::value) {
    return push_front<U>(merge<Comp>(Typelist<T, Ts...>(), Typelist<Us...>()));
  } else {
    return push_front<T>(merge<Comp>(Typelist<Ts...>(), Typelist<U, Us...>()));
  }
}

template <typename T, typename U, size_t N>
struct Split;

template <size_t... Indices, typename... Types, size_t N>
struct Split<std::index_sequence<Indices...>, Typelist<Types...>, N> {
  static constexpr auto getLeftHalf() {
    return (IntermediateTypelist<>() | ... |
            std::conditional_t < Indices<N, IntermediateTypelist<Types>, IntermediateTypelist<>>())
        .getTypelist();
  }
  static constexpr auto getRightHalf() {
    return (IntermediateTypelist<>() | ... |
            std::conditional_t<Indices >= N, IntermediateTypelist<Types>, IntermediateTypelist<>>())
        .getTypelist();
  }

  using left = decltype(getLeftHalf());
  using right = decltype(getRightHalf());
};

template <template <typename, typename> typename Comp>
constexpr Typelist<> stable_sort(Typelist<>) {
  return {};
}

template <template <typename, typename> typename Comp, typename T>
constexpr Typelist<T> stable_sort(Typelist<T>) {
  return {};
}

template <template <typename, typename> typename Comp, typename... Types>
constexpr auto stable_sort(Typelist<Types...>) {
  constexpr size_t n = sizeof...(Types);
  using CurrentSplit = Split<std::make_index_sequence<n>, Typelist<Types...>, n / 2>;
  return merge<Comp>(stable_sort<Comp>(typename CurrentSplit::left()),
                     stable_sort<Comp>(typename CurrentSplit::right()));
}

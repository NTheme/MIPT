//
// Created by Pavel Akhtyamov on 02.05.2020.
//

#include "WeatherTestCase.h"

#include "WeatherMock.h"

TEST(WeatherTestCase, NotFound) {
  WeatherMock weather;
  weather.DelegateToFake();
  EXPECT_CALL(weather, Get);
  EXPECT_THROW(
      {
        try {
          weather.GetResponseForCity("sd", "sd");
        } catch (std::invalid_argument& error) {
          EXPECT_STREQ("Api error. City is bad", error.what());
          throw std::invalid_argument("Api error. City is bad");
        }
      },
      std::invalid_argument);
}

TEST(WeatherTestCase, Found) {
  WeatherMock weather;
  weather.DelegateToFake();
  EXPECT_CALL(weather, Get).Times(4);
  EXPECT_EQ(weather.GetDifferenceString("Dubai", "Amsterdam"),
            "Weather in Dubai is warmer than in Amsterdam by 16 degrees");
  EXPECT_EQ(weather.GetDifferenceString("Amsterdam", "Dubai"),
            "Weather in Amsterdam is colder than in Dubai by 16 degrees");
}

TEST(WeatherTestCase, Diff) {
  WeatherMock weather;
  weather.DelegateToFake();
  EXPECT_CALL(weather, Get).Times(10);
  EXPECT_EQ(weather.GetTomorrowDiff("Dubai"),
            "The weather in Dubai tomorrow will be colder than today.");
  EXPECT_EQ(weather.GetTomorrowDiff("Moscow"),
            "The weather in Moscow tomorrow will be warmer than today.");
  EXPECT_EQ(weather.GetTomorrowDiff("Aljir"),
            "The weather in Aljir tomorrow will be much warmer than today.");
  EXPECT_EQ(weather.GetTomorrowDiff("Mexico"),
            "The weather in Mexico tomorrow will be the same than today.");
  EXPECT_EQ(weather.GetTomorrowDiff("Murmansk"),
            "The weather in Murmansk tomorrow will be much colder than today.");
}

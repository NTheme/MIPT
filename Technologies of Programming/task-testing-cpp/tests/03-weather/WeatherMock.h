//
// Created by Pavel Akhtyamov on 02.05.2020.
//

#pragma once

#include <Weather.h>
#include <gmock/gmock.h>

#include <map>

class WeatherFake : public Weather {
 public:
  WeatherFake();
  cpr::Response Get(const std::string& city, const cpr::Url& url) final;

 private:
  std::map<std::string, std::string> cities_;
};

class WeatherMock : public Weather {
 public:
  WeatherMock() = default;
  MOCK_METHOD(cpr::Response, Get,
              (const std::string& city, const cpr::Url& url), (override));

  void DelegateToFake();

 private:
  WeatherFake fake_;
};

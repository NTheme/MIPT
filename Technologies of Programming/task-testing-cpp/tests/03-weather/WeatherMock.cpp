//
// Created by Pavel Akhtyamov on 02.05.2020.
//

#include "WeatherMock.h"

WeatherFake::WeatherFake() {
  cities_["Dubai"] =
      R"({"list": [{"main": {"temp": 27.32},"dt_txt": "2023-04-10 21:00:00"},{"main": {"temp": 26.46},"dt_txt": "2023-04-11 00:00:00"},{"main": {"temp": 25.94},"dt_txt": "2023-04-11 03:00:00"},{"main": {"temp": 28.42},"dt_txt": "2023-04-11 06:00:00"},{"main": {"temp": 30.15},"dt_txt": "2023-04-11 09:00:00"},{"main": {"temp": 28.34},"dt_txt": "2023-04-11 12:00:00"},{"main": {"temp": 26.61},"dt_txt": "2023-04-11 15:00:00"},{"main": {"temp": 26.45},"dt_txt": "2023-04-11 18:00:00"},{"main": {"temp": 26.33},"dt_txt": "2023-04-11 21:00:00"}],"city": {"id": 292223,"name": "Dubai"}})";
  cities_["Amsterdam"] =
      R"({"list": [{"main": {"temp": 10.95},"dt_txt": "2023-04-10 21:00:00"},{"main": {"temp": 9.91},"dt_txt": "2023-04-11 00:00:00"},{"main": {"temp": 8.54},"dt_txt": "2023-04-11 03:00:00"},{"main": {"temp": 6.87},"dt_txt": "2023-04-11 06:00:00"},{"main": {"temp": 7.28},"dt_txt": "2023-04-11 09:00:00"},{"main": {"temp": 9.8},"dt_txt": "2023-04-11 12:00:00"},{"main": {"temp": 10.51},"dt_txt": "2023-04-11 15:00:00"},{"main": {"temp": 8.69},"dt_txt": "2023-04-11 18:00:00"},{"main": {"temp": 8.19},"dt_txt": "2023-04-11 21:00:00"}],"city": {"id": 2759794,"name": "Amsterdam"}})";
  cities_["Moscow"] =
      R"({"list": [{"main": {"temp": 6.51},"dt_txt": "2023-04-11 00:00:00"},{"main": {"temp": 6.737},"dt_txt": "2023-04-11 03:00:00"},{"main": {"temp": 9.66},"dt_txt": "2023-04-11 06:00:00"},{"main": {"temp": 13.31},"dt_txt": "2023-04-11 09:00:00"},{"main": {"temp": 14.58},"dt_txt": "2023-04-11 12:00:00"},{"main": {"temp": 14.17},"dt_txt": "2023-04-11 15:00:00"},{"main": {"temp": 12.16},"dt_txt": "2023-04-11 18:00:00"},{"main": {"temp": 10.12},"dt_txt": "2023-04-11 21:00:00"},{"main": {"temp": 8.42},"dt_txt": "2023-04-12 00:00:00"}],"city": {"id": 524901,"name": "Moscow"}})";
  cities_["Aljir"] =
      R"({"list": [{"main": {"temp": 13.074},"dt_txt": "2023-04-11 00:00:00"},{"main": {"temp": 13.711},"dt_txt": "2023-04-11 03:00:00"},{"main": {"temp": 14.93},"dt_txt": "2023-04-11 06:00:00"},{"main": {"temp": 18.91},"dt_txt": "2023-04-11 09:00:00"},{"main": {"temp": 21.6},"dt_txt": "2023-04-11 12:00:00"},{"main": {"temp": 21.2},"dt_txt": "2023-04-11 15:00:00"},{"main": {"temp": 18.77},"dt_txt": "2023-04-11 18:00:00"},{"main": {"temp": 18.47},"dt_txt": "2023-04-11 21:00:00"},{"main": {"temp": 17.95},"dt_txt": "2023-04-12 00:00:00"}],"city": {"id": 2507480,"name": "Algiers"}})";
  cities_["Mexico"] =
      R"({"list": [{"main": {"temp": 27.238},"dt_txt": "2023-04-11 00:00:00"},{"main": {"temp": 31.356},"dt_txt": "2023-04-11 03:00:00"},{"main": {"temp": 33.58},"dt_txt": "2023-04-11 06:00:00"},{"main": {"temp": 30.52},"dt_txt": "2023-04-11 09:00:00"},{"main": {"temp": 26.99},"dt_txt": "2023-04-11 12:00:00"},{"main": {"temp": 25.39},"dt_txt": "2023-04-11 15:00:00"},{"main": {"temp": 24.27},"dt_txt": "2023-04-11 18:00:00"},{"main": {"temp": 23.61},"dt_txt": "2023-04-11 21:00:00"},{"main": {"temp": 27.29},"dt_txt": "2023-04-12 00:00:00"}],"city": {"id": 1699805,"name": "Mexico"}})";
  cities_["Murmansk"] =
      R"({"list": [{"main": {"temp": -0.45},"dt_txt": "2023-04-11 00:00:00"},{"main": {"temp": -0.81},"dt_txt": "2023-04-11 03:00:00"},{"main": {"temp": 0.59},"dt_txt": "2023-04-11 06:00:00"},{"main": {"temp": 1.29},"dt_txt": "2023-04-11 09:00:00"},{"main": {"temp": 1.71},"dt_txt": "2023-04-11 12:00:00"},{"main": {"temp": 1.35},"dt_txt": "2023-04-11 15:00:00"},{"main": {"temp": -0.01},"dt_txt": "2023-04-11 18:00:00"},{"main": {"temp": -0.39},"dt_txt": "2023-04-11 21:00:00"},{"main": {"temp": -4.45},"dt_txt": "2023-04-12 00:00:00"}],"city": {"id": 524304,"name": "Murmansk Oblast"}})";
}

cpr::Response WeatherFake::Get(const std::string& city, const cpr::Url& url) {
  cpr::Response response;
  response.status_code = (cities_.count(city) == 1) ? 200 : 404;
  response.text = cities_[city];
  return response;
}

void WeatherMock::DelegateToFake() {
  ON_CALL(*this, Get)
      .WillByDefault([this](const std::string& city, const cpr::Url& url) {
        return fake_.Get(city, url);
      });
}

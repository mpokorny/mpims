#include <cassert>
#include <string>
#include <unordered_map>

#include <MSColumns.h>

using namespace mpims;

const std::string&
mpims::mscol_nickname(MSColumns col) {
  static std::string time_nick("TIM");
  static std::string subtime_nick("SBT");
  static std::string field_nick("FLD");
  static std::string spectral_window_nick("SPW");
  static std::string baseline_nick("BAL");
  static std::string channel_nick("CH");
  static std::string polarization_product_nick("POL");
  static std::string complex_nick("CPX");
  static std::string fail("");

  switch (col) {
  case MSColumns::time:
    return time_nick;
    break;
  case MSColumns::subtime:
    return subtime_nick;
    break;
  case MSColumns::field:
    return field_nick;
    break;
  case MSColumns::spectral_window:
    return spectral_window_nick;
    break;
  case MSColumns::baseline:
    return baseline_nick;
    break;
  case MSColumns::channel:
    return channel_nick;
    break;
  case MSColumns::polarization_product:
    return polarization_product_nick;
    break;
  case MSColumns::complex:
    return complex_nick;
    break;
  }
  assert(false);
  return fail;
}

MSColumns
mpims::mscol(const std::string& nick) {
  static std::unordered_map<std::string, MSColumns> map {
    {mscol_nickname(MSColumns::time), MSColumns::time},
    {mscol_nickname(MSColumns::subtime), MSColumns::subtime},
    {mscol_nickname(MSColumns::field), MSColumns::field},
    {mscol_nickname(MSColumns::spectral_window), MSColumns::spectral_window},
    {mscol_nickname(MSColumns::baseline), MSColumns::baseline},
    {mscol_nickname(MSColumns::channel), MSColumns::channel},
    {mscol_nickname(MSColumns::polarization_product),
        MSColumns::polarization_product},
    {mscol_nickname(MSColumns::complex), MSColumns::complex}
  };
  return map[nick];
}

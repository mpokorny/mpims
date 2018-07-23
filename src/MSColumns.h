#ifndef MS_COLUMNS_H_
#define MS_COLUMNS_H_

#include <string>

namespace mpims {

enum class MSColumns {
  time,
  subtime,
  field,
  spectral_window,
  baseline,
  channel,
  polarization_product,
  complex
};

const std::string&
mscol_nickname(MSColumns col);

MSColumns
mscol(const std::string& nick);

} // end namespace mpims

#endif // MS_COLUMNS_H_

// Local Variables:
// mode: c++
// c-basic-offset: 2
// fill-column: 80
// indent-tabs-mode: nil
// coding: utf-8
// End:

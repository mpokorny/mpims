/* -*- mode: c++; c-basic-offset: 2; indent-tabs-mode: nil; -*- */
#ifndef READER_MPI_STATE_H_
#define READER_MPI_STATE_H_

#include <mpi.h>

#include <mpims.h>

#include <array>
#include <cassert>
#include <cstring>
#include <memory>
#include <mutex>
#include <tuple>

#define MPI_FILE_GET_POSITION_WORKAROUND
#ifdef MPI_FILE_GET_POSITION_WORKAROUND
# include <cassert>
#endif

namespace mpims {

class MPIHandles {
public:

  MPIHandles()
    : comm(MPI_COMM_NULL)
    , info(MPI_INFO_NULL)
    , file(MPI_FILE_NULL) {
  }

  MPIHandles(::MPI_Comm comm_, ::MPI_Info info_, ::MPI_File file_)
    : comm(comm_)
    , info(info_)
    , file(file_) {
  }

  MPIHandles(const MPIHandles&) = delete;
  MPIHandles(MPIHandles&&) = delete;
  MPIHandles& operator=(const MPIHandles&) = delete;
  MPIHandles& operator=(MPIHandles&&) = delete;

  ::MPI_Comm comm;
  ::MPI_Info info;
  ::MPI_File file;

  ~MPIHandles() {
    if (file != MPI_FILE_NULL)
      mpi_call(::MPI_File_close, &file);
    if (info != MPI_INFO_NULL)
      mpi_call(::MPI_Info_free, &info);
    if (comm != MPI_COMM_NULL && comm != MPI_COMM_WORLD)
      mpi_call(::MPI_Comm_free, &comm);
  }

  void
  lock() const {
    m_mtx.lock();
  }

  void
  unlock() const {
    m_mtx.unlock();
  }

private:
  mutable std::recursive_mutex m_mtx;
};

class MPIState {

public:

  MPIState()
    : m_current(
      std::make_shared<MPIHandles>(
        MPI_COMM_NULL,
        MPI_INFO_NULL,
        MPI_FILE_NULL))
    , m_path(new std::string()) {
  }

  MPIState(
    ::MPI_Comm comm,
    ::MPI_Info info,
    ::MPI_File file,
    const std::string& path)
    : m_current(std::make_shared<MPIHandles>(comm, info, file))
    , m_path(std::make_shared<std::string>(path)) {
  }

  MPIState(const MPIState& other) {
    std::lock_guard<const MPIState> lock(other);
    derive_from(other);
  }

  MPIState&
  operator=(const MPIState& other) {
    std::lock_guard<MPIState> lock(*this);
    if (this != &other) {
      MPIState temp(other);
      swap(temp);
    }
    return *this;
  }

  MPIState(MPIState&& other)
    : m_latent(std::move(other).m_latent)
    , m_current(std::move(other).m_current)
    , m_disp(std::move(other).m_disp)
    , m_filetype(std::move(other).m_filetype)
    , m_datarep(std::move(other).m_datarep)
    , m_offset(std::move(other).m_offset)
    , m_path(std::move(other).m_path) {
  }

  MPIState&
  operator=(MPIState&& other) {
    std::lock_guard<MPIState> lock(*this);
    m_latent = std::move(other).m_latent;
    m_current = std::move(other).m_current;
    m_disp = std::move(other).m_disp;
    m_filetype = std::move(other).m_filetype;
    m_datarep = std::move(other).m_datarep;
    m_offset = std::move(other).m_offset;
    m_path = std::move(other).m_path;
    return *this;
  }

  std::shared_ptr<MPIHandles>
  handles() const {
    actualize();
    return m_current;
  }

  const std::string&
  path() const {
    return *m_path;
  }

  void
  lock() const {
    m_mtx.lock();
  }

  void
  unlock() const {
    m_mtx.unlock();
  }

  void
  swap(MPIState& other) {
    using std::swap;
    std::lock_guard<MPIState> lock1(*this);
    std::lock_guard<MPIState> lock2(other);
    swap(m_current, other.m_current);
    swap(m_latent, other.m_latent);
    swap(m_disp, other.m_disp);
    swap(m_filetype, other.m_filetype);
    swap(m_datarep, other.m_datarep);
    swap(m_offset, other.m_offset);
    swap(m_path, other.m_path);
  }

protected:

  void
  derive_from(const MPIState& other) const {
    m_path = other.m_path;
    if (other.m_current) {
      std::lock_guard<MPIHandles> lock(*other.m_current);
      if (other.m_current->file != MPI_FILE_NULL) {
        other.m_disp = std::make_shared<const ::MPI_Offset>();
        other.m_filetype = datatype();
        other.m_datarep = std::make_shared<const datarep_t>();
        ::MPI_Datatype etype;
        mpi_call(
          ::MPI_File_get_view,
          other.m_current->file,
          std::const_pointer_cast<::MPI_Offset>(other.m_disp).get(),
          &etype,
          std::const_pointer_cast<::MPI_Datatype>(other.m_filetype).get(),
          std::const_pointer_cast<datarep_t>(other.m_datarep)->data());
#ifndef MPI_FILE_GET_POSITION_WORKAROUND
        mpi_call(
          ::MPI_File_get_position,
          other.m_current->file,
          const_cast<::MPI_Offset *>(&(other.m_offset)));
#endif
      }
#ifndef MPI_FILE_GET_POSITION_WORKAROUND
      mpi_call(::MPI_Barrier, other.m_current->comm);
#endif
      m_current = other.m_current;
      m_latent.reset();
      other.m_latent = m_current;
      other.m_current.reset();
    } else {
      m_current.reset();
      m_latent = other.m_latent;
      m_offset = other.m_offset;
      m_disp = other.m_disp;
      m_filetype = other.m_filetype;
      m_datarep = other.m_datarep;
    }
  }

  void
  actualize() const {
    if (m_current)
      return;
    if (!m_latent) {
      m_current =
        std::make_shared<MPIHandles>(
          MPI_COMM_NULL,
          MPI_INFO_NULL,
          MPI_FILE_NULL);
      m_disp.reset();
      m_filetype.reset();
      m_datarep.reset();
      m_offset = 0;
    } else {
      ::MPI_Comm comm;
      ::MPI_Info info;
      ::MPI_File file;
      std::lock_guard<MPIHandles> lock(*m_latent);
      if (m_latent->comm != MPI_COMM_NULL) {
        mpi_call(::MPI_Comm_dup, m_latent->comm, &comm);
        mpi_call(::MPI_Comm_set_errhandler, comm, MPI_ERRORS_RETURN);
      } else {
        comm = MPI_COMM_NULL;
      }
      if (m_latent->info != MPI_INFO_NULL)
        mpi_call(::MPI_Info_dup, m_latent->info, &info);
      else
        info = MPI_INFO_NULL;
      if (m_latent->file != MPI_FILE_NULL) {
        mpi_call(
          ::MPI_File_open,
          comm,
          m_path->c_str(),
          MPI_MODE_RDONLY,
          info,
          &file);
        mpi_call(::MPI_File_set_errhandler, file, MPI_ERRORS_RETURN);
        if (m_disp) {
#ifdef MPI_FILE_GET_POSITION_WORKAROUND
          assert(false);
#endif
          mpi_call(
            ::MPI_File_set_view,
            file,
            *m_disp,
            MPI_CXX_FLOAT_COMPLEX,
            *m_filetype,
            m_datarep->data(),
            MPI_INFO_NULL);
          mpi_call(::MPI_File_seek, file, m_offset, MPI_SEEK_SET);
        }
      } else {
        file = MPI_FILE_NULL;
      }
      m_latent.reset();
      m_current = std::make_shared<MPIHandles>(comm, info, file);
    }
  }

private:

  mutable std::recursive_mutex m_mtx;

  mutable std::shared_ptr<MPIHandles> m_latent;

  mutable std::shared_ptr<MPIHandles> m_current;

  mutable std::shared_ptr<const ::MPI_Offset> m_disp;

  mutable std::shared_ptr<const ::MPI_Datatype> m_filetype;

  typedef std::array<char, MPI_MAX_DATAREP_STRING> datarep_t;

  mutable std::shared_ptr<const datarep_t> m_datarep;

  mutable ::MPI_Offset m_offset;

  mutable std::shared_ptr<const std::string> m_path;
};

} // end namespace mpims

#endif // READER_MPI_STATE_H_

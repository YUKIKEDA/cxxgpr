// SPDX-License-Identifier: Apache-2.0
#include "cxxgpr/linalg/cholesky.hpp"

#include "cxxgpr/core/checked.hpp"
#include "cxxgpr/linalg/dense.hpp"
#include "linalg/eigen_map.hpp"

#include <Eigen/Cholesky>
#include <cmath>
#include <cstddef>
#include <vector>

namespace cxxgpr::linalg {

namespace {

template <KernelScalarConcept T>
[[nodiscard]] std::expected<void, GprError> validate_lower_factor(MatrixRef<T> l) {
    if (const auto check = detail::validate_matrix_view(l, true); !check.has_value()) {
        return check;
    }
    if (const auto fin = detail::require_lower_finite(l); !fin.has_value()) {
        return fin;
    }
    const std::size_t n = l.rows();
    for (std::size_t i = 0; i < n; ++i) {
        if (l(i, i) <= T{0}) {
            return std::unexpected(GprError::non_positive_definite_matrix());
        }
    }
    return {};
}

template <KernelScalarConcept T>
[[nodiscard]] std::expected<void, GprError> validate_solve_inputs(MatrixRef<T> l,
                                                                  MatrixMut<T> rhs) {
    if (const auto l_check = validate_lower_factor(l); !l_check.has_value()) {
        return l_check;
    }
    if (const auto rhs_check = detail::validate_matrix_view(rhs, false); !rhs_check.has_value()) {
        return rhs_check;
    }
    if (rhs.rows() != l.rows()) {
        return std::unexpected(
            GprError::length_mismatch("right-hand side row count must match factor dimension"));
    }
    if (const auto fin = detail::require_matrix_finite(detail::as_ref(rhs)); !fin.has_value()) {
        return fin;
    }
    return {};
}

template <KernelScalarConcept T>
[[nodiscard]] std::expected<void, GprError> run_eigen_llt_in_place(MatrixMut<T> a,
                                                                   CholeskyStage stage) {
    auto map = detail::to_eigen_map(a);
    using RefType = Eigen::Ref<detail::ColMajorDynamic<T>, 0, Eigen::OuterStride<>>;
    Eigen::LLT<RefType, Eigen::Lower> llt(map);
    const std::size_t n = a.rows();
    if (llt.info() != Eigen::Success) {
        return std::unexpected(GprError::cholesky_failed(0.0, n, stage));
    }
    for (std::size_t i = 0; i < n; ++i) {
        const T diag = a(i, i);
        if (!std::isfinite(diag) || diag <= T{0}) {
            return std::unexpected(GprError::cholesky_failed(0.0, n, stage));
        }
    }
    return {};
}

} // namespace

template <KernelScalarConcept T>
std::expected<void, GprError> cholesky_lower_in_place(MatrixMut<T> a, CholeskyStage stage) {
    if (const auto check = detail::validate_matrix_view(a, true); !check.has_value()) {
        return check;
    }
    if (const auto fin = detail::require_lower_finite(detail::as_ref(a)); !fin.has_value()) {
        return fin;
    }
    return run_eigen_llt_in_place(a, stage);
}

template <KernelScalarConcept T>
std::expected<double, GprError> cholesky_lower_with_jitter(MatrixMut<T> a, MatrixMut<T> backup,
                                                           const JitterPolicy& policy,
                                                           CholeskyStage stage) {
    if (const auto check = detail::validate_matrix_view(a, true); !check.has_value()) {
        return std::unexpected(check.error());
    }
    if (const auto check = detail::validate_matrix_view(backup, true); !check.has_value()) {
        return std::unexpected(check.error());
    }
    if (a.rows() != backup.rows()) {
        return std::unexpected(
            GprError::length_mismatch("backup matrix dimension must match target matrix"));
    }
    if (const auto fin = detail::require_lower_finite(detail::as_ref(a)); !fin.has_value()) {
        return std::unexpected(fin.error());
    }

    const std::size_t n = a.rows();
    if (const auto cp = copy_lower_triangle(detail::as_ref(a), backup); !cp.has_value()) {
        return std::unexpected(cp.error());
    }

    const auto first = run_eigen_llt_in_place(a, stage);
    if (first.has_value()) {
        return 0.0;
    }
    if (first.error().code() != GprErrorCode::CholeskyFailed) {
        return std::unexpected(first.error());
    }

    double last_j = 0.0;
    JitterSchedule schedule = policy.retry_schedule();
    while (const auto next_j = schedule.next()) {
        const double j = *next_j;
        last_j = j;
        if (const auto restore = copy_lower_triangle(detail::as_ref(backup), a);
            !restore.has_value()) {
            return std::unexpected(restore.error());
        }
        if (const auto add = add_to_diag(a, static_cast<T>(j)); !add.has_value()) {
            return std::unexpected(add.error());
        }
        const auto attempt = run_eigen_llt_in_place(a, stage);
        if (attempt.has_value()) {
            return j;
        }
        if (attempt.error().code() != GprErrorCode::CholeskyFailed) {
            return std::unexpected(attempt.error());
        }
    }

    return std::unexpected(GprError::cholesky_failed(last_j, n, stage));
}

template <KernelScalarConcept T>
std::expected<double, GprError>
cholesky_lower_with_jitter(MatrixMut<T> a, const JitterPolicy& policy, CholeskyStage stage) {
    if (const auto check = detail::validate_matrix_view(a, true); !check.has_value()) {
        return std::unexpected(check.error());
    }
    const auto len = column_major_len(a.rows(), a.cols());
    if (!len.has_value()) {
        return std::unexpected(len.error());
    }
    std::vector<T> backup_storage(*len);
    MatrixMut<T> backup(backup_storage.data(), a.rows(), a.cols());
    return cholesky_lower_with_jitter(a, backup, policy, stage);
}

template <KernelScalarConcept T>
std::expected<void, GprError> solve_lower_in_place(MatrixRef<T> l, MatrixMut<T> rhs) {
    if (const auto check = validate_solve_inputs(l, rhs); !check.has_value()) {
        return check;
    }
    const auto l_map = detail::to_eigen_map(l);
    auto rhs_map = detail::to_eigen_map(rhs);
    l_map.template triangularView<Eigen::Lower>().solveInPlace(rhs_map);
    return {};
}

template <KernelScalarConcept T>
std::expected<void, GprError> solve_lower_in_place(MatrixRef<T> l, std::span<T> rhs) {
    if (rhs.empty()) {
        return std::unexpected(GprError::empty_input());
    }
    return solve_lower_in_place(l, MatrixMut<T>(rhs.data(), rhs.size(), 1));
}

template <KernelScalarConcept T>
std::expected<void, GprError> solve_lower_transpose_in_place(MatrixRef<T> l, MatrixMut<T> rhs) {
    if (const auto check = validate_solve_inputs(l, rhs); !check.has_value()) {
        return check;
    }
    const auto l_map = detail::to_eigen_map(l);
    auto rhs_map = detail::to_eigen_map(rhs);
    l_map.transpose().template triangularView<Eigen::Upper>().solveInPlace(rhs_map);
    return {};
}

template <KernelScalarConcept T>
std::expected<void, GprError> solve_lower_transpose_in_place(MatrixRef<T> l, std::span<T> rhs) {
    if (rhs.empty()) {
        return std::unexpected(GprError::empty_input());
    }
    return solve_lower_transpose_in_place(l, MatrixMut<T>(rhs.data(), rhs.size(), 1));
}

template <KernelScalarConcept T>
std::expected<void, GprError> solve_llt_in_place(MatrixRef<T> l, MatrixMut<T> rhs) {
    if (const auto check = validate_solve_inputs(l, rhs); !check.has_value()) {
        return check;
    }
    const auto l_map = detail::to_eigen_map(l);
    auto rhs_map = detail::to_eigen_map(rhs);
    l_map.template triangularView<Eigen::Lower>().solveInPlace(rhs_map);
    l_map.transpose().template triangularView<Eigen::Upper>().solveInPlace(rhs_map);
    return {};
}

template <KernelScalarConcept T>
std::expected<void, GprError> solve_llt_in_place(MatrixRef<T> l, std::span<T> rhs) {
    if (rhs.empty()) {
        return std::unexpected(GprError::empty_input());
    }
    return solve_llt_in_place(l, MatrixMut<T>(rhs.data(), rhs.size(), 1));
}

template <KernelScalarConcept T>
std::expected<void, GprError> inv_diag_from_chol_l(MatrixRef<T> l, std::span<T> out_diag,
                                                   MatrixMut<T> scratch_n_by_n) {
    if (const auto l_check = validate_lower_factor(l); !l_check.has_value()) {
        return l_check;
    }
    if (out_diag.empty()) {
        return std::unexpected(GprError::empty_input());
    }
    if (const auto cnt = require_count(out_diag.size(), l.rows(), "inv_diag output length");
        !cnt.has_value()) {
        return cnt;
    }
    if (const auto sc_check = detail::validate_matrix_view(scratch_n_by_n, true);
        !sc_check.has_value()) {
        return sc_check;
    }
    if (scratch_n_by_n.rows() != l.rows()) {
        return std::unexpected(
            GprError::length_mismatch("scratch matrix dimension must match factor dimension"));
    }

    const std::size_t n = l.rows();
    for (std::size_t col = 0; col < n; ++col) {
        for (std::size_t row = 0; row < n; ++row) {
            scratch_n_by_n(row, col) = (row == col) ? T{1} : T{0};
        }
    }

    const auto l_map = detail::to_eigen_map(l);
    auto scratch_map = detail::to_eigen_map(scratch_n_by_n);
    l_map.template triangularView<Eigen::Lower>().solveInPlace(scratch_map);

    for (std::size_t col = 0; col < n; ++col) {
        double sum_sq = 0.0;
        for (std::size_t row = col; row < n; ++row) {
            const auto val = static_cast<double>(scratch_n_by_n(row, col));
            sum_sq += val * val;
        }
        out_diag[col] = static_cast<T>(sum_sq);
    }
    return {};
}

template <KernelScalarConcept T> std::expected<double, GprError> log_det_from_l(MatrixRef<T> l) {
    if (const auto check = validate_lower_factor(l); !check.has_value()) {
        return std::unexpected(check.error());
    }
    const std::size_t n = l.rows();
    double sum = 0.0;
    double comp = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        const double term = 2.0 * std::log(static_cast<double>(l(i, i)));
        const double t = sum + term;
        if (std::abs(sum) >= std::abs(term)) {
            comp += (sum - t) + term;
        } else {
            comp += (term - t) + sum;
        }
        sum = t;
    }
    return sum + comp;
}

template <KernelScalarConcept T>
std::expected<double, GprError> half_log_det_from_l(MatrixRef<T> l) {
    const auto ld = log_det_from_l(l);
    if (!ld.has_value()) {
        return std::unexpected(ld.error());
    }
    return 0.5 * (*ld);
}

template std::expected<void, GprError> cholesky_lower_in_place<float>(MatrixMut<float>,
                                                                      CholeskyStage);
template std::expected<void, GprError> cholesky_lower_in_place<double>(MatrixMut<double>,
                                                                       CholeskyStage);

template std::expected<double, GprError> cholesky_lower_with_jitter<float>(MatrixMut<float>,
                                                                           MatrixMut<float>,
                                                                           const JitterPolicy&,
                                                                           CholeskyStage);
template std::expected<double, GprError> cholesky_lower_with_jitter<double>(MatrixMut<double>,
                                                                            MatrixMut<double>,
                                                                            const JitterPolicy&,
                                                                            CholeskyStage);

template std::expected<double, GprError>
cholesky_lower_with_jitter<float>(MatrixMut<float>, const JitterPolicy&, CholeskyStage);
template std::expected<double, GprError>
cholesky_lower_with_jitter<double>(MatrixMut<double>, const JitterPolicy&, CholeskyStage);

template std::expected<void, GprError> solve_lower_in_place<float>(MatrixRef<float>,
                                                                   MatrixMut<float>);
template std::expected<void, GprError> solve_lower_in_place<double>(MatrixRef<double>,
                                                                    MatrixMut<double>);

template std::expected<void, GprError> solve_lower_in_place<float>(MatrixRef<float>,
                                                                   std::span<float>);
template std::expected<void, GprError> solve_lower_in_place<double>(MatrixRef<double>,
                                                                    std::span<double>);

template std::expected<void, GprError> solve_lower_transpose_in_place<float>(MatrixRef<float>,
                                                                             MatrixMut<float>);
template std::expected<void, GprError> solve_lower_transpose_in_place<double>(MatrixRef<double>,
                                                                              MatrixMut<double>);

template std::expected<void, GprError> solve_lower_transpose_in_place<float>(MatrixRef<float>,
                                                                             std::span<float>);
template std::expected<void, GprError> solve_lower_transpose_in_place<double>(MatrixRef<double>,
                                                                              std::span<double>);

template std::expected<void, GprError> solve_llt_in_place<float>(MatrixRef<float>,
                                                                 MatrixMut<float>);
template std::expected<void, GprError> solve_llt_in_place<double>(MatrixRef<double>,
                                                                  MatrixMut<double>);

template std::expected<void, GprError> solve_llt_in_place<float>(MatrixRef<float>,
                                                                 std::span<float>);
template std::expected<void, GprError> solve_llt_in_place<double>(MatrixRef<double>,
                                                                  std::span<double>);

template std::expected<void, GprError>
    inv_diag_from_chol_l<float>(MatrixRef<float>, std::span<float>, MatrixMut<float>);
template std::expected<void, GprError>
    inv_diag_from_chol_l<double>(MatrixRef<double>, std::span<double>, MatrixMut<double>);

template std::expected<double, GprError> log_det_from_l<float>(MatrixRef<float>);
template std::expected<double, GprError> log_det_from_l<double>(MatrixRef<double>);

template std::expected<double, GprError> half_log_det_from_l<float>(MatrixRef<float>);
template std::expected<double, GprError> half_log_det_from_l<double>(MatrixRef<double>);

} // namespace cxxgpr::linalg

// SPDX-License-Identifier: Apache-2.0
// How-to: save a model trained on supplied squared distances, load it back, and bind new tables to the loaded kernel.
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cxxgpr/gpr/distance_gpr.hpp>
#include <cxxgpr/kernel/kernel.hpp>
#include <cxxgpr/persist/persist.hpp>
#include <filesystem>
#include <iostream>
#include <numbers>
#include <vector>

namespace
{
    // Squared arc length between two angles on the unit circle.
    double squared_arc(double a, double b)
    {
        const double two_pi = 2.0 * std::numbers::pi;
        double diff = std::fmod(std::abs(a - b), two_pi);
        diff = std::min(diff, two_pi - diff);
        return diff * diff;
    }

    // Column-major table of squared arc lengths between two sets of angles.
    std::vector<double> arc_table(const std::vector<double>& rows, const std::vector<double>& cols)
    {
        std::vector<double> table(rows.size() * cols.size());
        for (std::size_t j = 0; j < cols.size(); ++j)
        {
            for (std::size_t i = 0; i < rows.size(); ++i)
            {
                table[(j * rows.size()) + i] = squared_arc(rows[i], cols[j]);
            }
        }
        return table;
    }
}  // namespace

int main()
{
    constexpr std::size_t n = 16;
    std::vector<double> angle(n);
    std::vector<double> y(n);
    for (std::size_t i = 0; i < n; ++i)
    {
        angle[i] = 2.0 * std::numbers::pi * static_cast<double>(i) / static_cast<double>(n);
        y[i] = std::sin(angle[i]) + (0.3 * std::cos(2.0 * angle[i]));
    }
    const std::vector<double> query_angle{0.4, 2.2, 5.1};
    const std::size_t m = query_angle.size();

    const cxxgpr::ScalarDistance arc_slot;
    const auto kernel = cxxgpr::DistanceKernel<cxxgpr::DistanceOnly>(cxxgpr::ConstantKernel::create(1.0).value())
        * arc_slot.matern(cxxgpr::MaternKernel::create(1.0, cxxgpr::MaternNu::OneHalf).value());
    const std::vector<double> train_table = arc_table(angle, angle);
    const std::vector<double> cross_table = arc_table(angle, query_angle);
    cxxgpr::DistanceSource train;
    train.bind_scalar(arc_slot.slot(), cxxgpr::MatrixRef<double>(train_table.data(), n, n));
    cxxgpr::DistanceSource cross;
    cross.bind_scalar(arc_slot.slot(), cxxgpr::MatrixRef<double>(cross_table.data(), n, m));

    auto trainer = cxxgpr::DistanceGpr<cxxgpr::optimizer::Lbfgs>(kernel).with_noise_variance(0.01);
    if (!trainer)
    {
        std::cerr << trainer.error().message() << '\n';
        return 1;
    }
    auto model = trainer->fit(train, y);
    if (!model)
    {
        std::cerr << model.error().message() << '\n';
        return 1;
    }
    auto before = model->predict(cross);
    if (!before)
    {
        std::cerr << before.error().message() << '\n';
        return 1;
    }

    // [save]
    // The checkpoint holds the training distances (one packed table per slot), the targets, and optionally the factor.
    const std::filesystem::path dir = std::filesystem::temp_directory_path() / "cxxgpr_example_distance_model";
    if (auto saved = cxxgpr::persist::save_with_factor(*model, dir); !saved)
    {
        std::cerr << saved.error().message() << '\n';
        return 1;
    }
    // [end]

    // [load]
    // The mode must match the one the model was saved with: DistanceOnly here.
    auto loaded = cxxgpr::persist::LoadedDistanceGpr<cxxgpr::DistanceOnly>::load(dir);
    if (!loaded)
    {
        std::cerr << loaded.error().message() << '\n';
        return 1;
    }
    // [end]

    // [bind]
    // Loading issues fresh slots. Take them from the loaded kernel to bind the query tables.
    const auto slots = loaded->kernel().slots();
    cxxgpr::DistanceSource loaded_cross;
    loaded_cross.bind_scalar(slots[0].id, cxxgpr::MatrixRef<double>(cross_table.data(), n, m));
    auto after = loaded->predict(loaded_cross);
    // [end]
    if (!after)
    {
        std::cerr << after.error().message() << '\n';
        return 1;
    }

    // The slot used before saving belongs to another kernel and is refused.
    const auto stale = loaded->predict(cross);
    // [wrong-loader]
    // A with-points loader refuses a distance-only checkpoint.
    const auto wrong = cxxgpr::persist::LoadedDistanceGpr<cxxgpr::WithPoints>::load(dir);
    // [end]

    bool ok = !stale.has_value() && !wrong.has_value()
        && wrong.error().persist_error_kind() == cxxgpr::PersistErrorKind::WrongModel;
    for (std::size_t j = 0; j < m; ++j)
    {
        std::cout << "angle " << query_angle[j] << ": before = " << before->mean[j] << ", after = " << after->mean[j]
                  << '\n';
        ok = ok && before->mean[j] == after->mean[j] && before->variance[j] == after->variance[j];
    }
    std::error_code ec;
    std::filesystem::remove_all(dir, ec);
    return ok ? 0 : 1;
}

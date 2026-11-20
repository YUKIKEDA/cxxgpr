// SPDX-License-Identifier: Apache-2.0
// Inspect the build provenance of the cxxgpr library you linked against.
#include <cxxgpr/core/build_info.hpp>

#include <iostream>

int main()
{
    // [struct]
    const cxxgpr::BuildInfo info = cxxgpr::build_info();
    std::cout << "schema:        " << info.schema << '\n'
              << "version:       " << info.version << '\n'
              << "git commit:    " << info.git_commit << (info.uncommitted_changes ? " (dirty)" : "") << '\n'
              << "compiler:      " << info.compiler << '\n'
              << "platform:      " << info.platform << '\n'
              << "platform key:  " << info.platform_key << '\n'
              << "configuration: " << info.configuration << '\n'
              << "fp mode:      " << info.floating_point << '\n';
    for (const cxxgpr::DependencyVersion& dep : info.dependencies)
    {
        std::cout << "dependency:    " << dep.name << ' ' << dep.version << '\n';
    }
    // [end]

    // [json]
    // The same data as one JSON document, handy for logs and bug reports.
    const std::string json = cxxgpr::build_info_json();
    std::cout << '\n' << json;
    // [end]

    // The program fails if the library reports no version or the JSON does not carry it.
    const bool ok = !info.version.empty() && json.find(info.version) != std::string::npos;
    return ok ? 0 : 1;
}

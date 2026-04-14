// SPDX-FileCopyrightText: © 2025 PRISMS Center at the University of Michigan
// SPDX-License-Identifier: GNU Lesser General Public Version 2.1

#include "custom_pde.h"

#include <prismspf/core/conditional_ostreams.h>
#include <prismspf/core/parse_cmd_options.h>
#include <prismspf/core/problem.h>
#include <prismspf/core/solve_block.h>

using namespace prisms;

int
main(int argc, char *argv[])
{
  // Initialize MPI
  prisms::MPI_InitFinalize mpi_init(argc, argv);

  // Restrict deal.II console printing
  dealii::deallog.depth_console(0);

  // Parse the command line options (if there are any) to get the name of the
  // input file
  ParseCMDOptions cli_options(argc, argv);

  constexpr unsigned int dim    = 2;
  constexpr unsigned int degree = 1;

  std::vector<FieldAttributes> fields = {
    FieldAttributes("u"), //
    FieldAttributes("v"), //
  };

  SolveBlock main_fields(0, Explicit, Initialized, {0, 1});
  main_fields.dependencies_rhs =
    make_dependency_set(fields,
                        {"old_1(u)", "old_1(v)", "grad(old_1(u))", "grad(old_1(v))"});

  std::vector<SolveBlock> solve_blocks({main_fields});

  UserInputParameters<dim>       user_inputs(cli_options.get_parameters_filename());
  PhaseFieldTools<dim>           pf_tools;
  GrayScott<dim, degree, double> pde_operator(user_inputs, pf_tools);
  if (pde_operator.auto_dt)
    {
      // assuming square and no subdivisions
      double dx = user_inputs.spatial_discretization.rectangular_mesh.size[0] /
                  (1 << user_inputs.spatial_discretization.global_refinement);
      double D         = std::max(pde_operator.Du, pde_operator.Dv);
      double stability = pde_operator.stability;
      user_inputs.temporal_discretization.dt =
        stability * (dx * dx) / (2 * dim * D * degree * degree);
      ConditionalOStreams::pout_base()
        << "Time step size (dt) set to: " << user_inputs.temporal_discretization.dt
        << std::endl;
    }

  Problem<dim, degree, double> problem(fields,
                                       solve_blocks,
                                       user_inputs,
                                       pf_tools,
                                       pde_operator);
  problem.solve();

  return 0;
}

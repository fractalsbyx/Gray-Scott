// SPDX-FileCopyrightText: © 2025 Xander Mensah
// SPDX-License-Identifier: GNU Lesser General Public Version 2.1

#include <prismspf/core/pde_operator_base.h>

#include <cmath>

PRISMS_PF_BEGIN_NAMESPACE

template <unsigned int dim, unsigned int degree, typename number>
class GrayScott : public PDEOperatorBase<dim, degree, number>
{
public:
  using ScalarValue = dealii::VectorizedArray<number>;
  using ScalarGrad  = dealii::Tensor<1, dim, ScalarValue>;
  using ScalarHess  = dealii::Tensor<2, dim, ScalarValue>;

  using VectorValue = dealii::Tensor<1, dim, ScalarValue>;
  using VectorGrad  = dealii::Tensor<2, dim, ScalarValue>;
  using VectorHess  = dealii::Tensor<3, dim, ScalarValue>;

  using PDEOperatorBase<dim, degree, number>::get_user_inputs;
  using PDEOperatorBase<dim, degree, number>::get_pf_tools;

  number Du = 0.00002;
  number Dv = 0.00001;
  number f  = 0.04;
  number k  = 0.06;

  double stability = 1.0;  // Stability factor (less than 1 for stability)
  bool   auto_dt   = true; // Whether to automatically compute dt based on stability

  /**
   * @brief Constructor.
   */
  explicit GrayScott(const UserInputParameters<dim> &_user_inputs,
                     PhaseFieldTools<dim>           &_pf_tools)
    : PDEOperatorBase<dim, degree, number>(_user_inputs, _pf_tools)
    , Du(_user_inputs.user_constants.get_double("Du"))
    , Dv(_user_inputs.user_constants.get_double("Dv"))
    , f(_user_inputs.user_constants.get_double("f"))
    , k(_user_inputs.user_constants.get_double("k"))
    , stability(_user_inputs.user_constants.get_double("stability"))
    , auto_dt(_user_inputs.user_constants.get_bool("auto_dt"))
  {}

private:
  void
  set_initial_condition([[maybe_unused]] const unsigned int       &index,
                        [[maybe_unused]] const unsigned int       &component,
                        [[maybe_unused]] const dealii::Point<dim> &point,
                        [[maybe_unused]] number                   &scalar_value,
                        [[maybe_unused]] number &vector_component_value) const override
  {
    using std::cos;
    using std::max;
    using std::min;
    using std::sin;
    using std::sqrt;
    using std::tanh;
    // Custom coordinate system
    const dealii::Tensor<1, dim> &mesh_size =
      get_user_inputs().spatial_discretization.rectangular_mesh.size;
    const dealii::Point<dim>      center(mesh_size / 2.0);
    const dealii::Point<dim>      p(point - center);
    [[maybe_unused]] const double x  = (dim > 0) ? p[0] : 0.;
    [[maybe_unused]] const double y  = (dim > 1) ? p[1] : 0.;
    [[maybe_unused]] const double z  = (dim > 2) ? p[2] : 0.;
    [[maybe_unused]] const double r2 = x * x + y * y + z * z;

    // ===========================================================================
    // FUNCTION FOR INITIAL CONDITIONS
    // ===========================================================================
    double w   = 0.05;
    double rad = 0.3;
    double u = 0.5 * (tanh(1.0 * sin(10.0 * x) - (rad * rad - r2) / (2 * rad * w)) + 1.0);
    double v = 1.0 - u;
    if (index == 0)
      {
        scalar_value = u;
        return;
      }
    if (index == 1)
      {
        scalar_value = v;
        return;
      }
  }

  void
  compute_rhs([[maybe_unused]] FieldContainer<dim, degree, number> &variable_list,
              [[maybe_unused]] const SimulationTimer               &sim_timer,
              [[maybe_unused]] unsigned int solve_block_id) const override
  {
    const number dt = sim_timer.get_timestep();
    if (solve_block_id == 0) // n, x
      {
        const ScalarValue u_val  = variable_list.template get_value<Scalar, OldOne>(0);
        const ScalarGrad  u_grad = variable_list.template get_gradient<Scalar, OldOne>(0);
        const ScalarValue v_val  = variable_list.template get_value<Scalar, OldOne>(1);
        const ScalarGrad  v_grad = variable_list.template get_gradient<Scalar, OldOne>(1);

        const ScalarValue dudt_val = -u_val * v_val * v_val + f * (1.0 - u_val);
        const ScalarValue dvdt_val = +u_val * v_val * v_val - (f + k) * v_val;
        const VectorValue dudt_vec = Du * u_grad;
        const VectorValue dvdt_vec = Dv * v_grad;

        // u
        variable_list.set_value_term(0, u_val + dudt_val * dt);
        variable_list.set_gradient_term(0, -dudt_vec * dt);

        // v
        variable_list.set_value_term(1, v_val + dvdt_val * dt);
        variable_list.set_gradient_term(1, -dvdt_vec * dt);
      }
  }
};

PRISMS_PF_END_NAMESPACE

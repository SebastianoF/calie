

def lie_log_composition():
    # TODO
    pass


def lie_exp_sum():
    # TODO
    pass


def lie_exp_multiplication():
    # TODO
    pass



'''

@classmethod
    def log_composition(cls, svf_im0, svf_im1, kind=('ground',), answer='dom'):
        """
        From two stationary velocity fields in the tangent space he returns their composition in the Lie group.
        No log is computed on the resulting def (deformation field)
        :param svf_im0:
        :param svf_im1:
        :param kind: 'ground'
            ground truth for the composition, it returns the value \exp(svf_im0)\circ \exp(svf_im1)
        :param answer:
        # TODO select if you want the answer in the domain or not!

        kind: 'bch0'
            composition using the bch 0: it returns
            \exp(svf_im0)\circ \exp(svf_im1) \simeq \exp(svf_im0) + \exp(svf_im1)
        kind: 'bch1'
            composition using the bch 1: it returns
            \exp(svf_im0)\circ \exp(svf_im1) \simeq \exp(svf_im0) + \exp(svf_im1) + .5 [\exp(svf_im0), \exp(svf_im1)]
        kind: 'pt'
            composition using parallel transport: it returns
            \exp(svf_im0)\circ \exp(svf_im1) \simeq ...
        kind: 'pt_warp'
            composition using parallel transport warped: it returns
            \exp(svf_im0)\circ \exp(svf_im1) \simeq ...

        kind: numerical closed, if len = 2 - [kind_log, kind_exp], uses a numerical method for the
            log and a numerical method for the exp, in the formula exp(log(svf_im0) o log(svf_im1))

        answer: 'svf' returns approximation of log(exp(svf_im0) o exp(svf_im1))
             or 'def' returns approximation of exp(svf_im0) o exp(svf_im1), more manageable to compute the ground truth.

        :return: exp(svf_im0) o exp(svf_im1) or log(exp(svf_im0) o exp(svf_im1)) according to the flag answer.
        It is in the Lie group and not in the Lie algebra, since the computation of the error is in the Lie group.
        """
        # TODO: refactor this code within debugging!

        str_error_kind = 'Error: wrong input data for the chosen kind of composition.'
        str_error_answer = 'Error: wrong input data for the chosen answer of composition.'

        if len(kind) == 1:
            # Numerical methods BCH based and related

            if kind == 'bch0':
                svf_bch0 = copy.deepcopy(svf_im0)
                svf_bch0.field.data += svf_im1.field.data
                def_result = svf_bch0.exponential()

            elif kind == 'bch1':
                vel_bch1 = copy.deepcopy(svf_im0)
                vel_bch1.field.data += svf_im1.field.data
                vel_bch1.field.data += 0.5 * svf_im0.lie_bracket(svf_im1).field.data
                def_result = vel_bch1.exponential()

            elif kind == 'bch1.5':
                vel_bch2 = copy.deepcopy(svf_im0)
                vel_bch2.field.data += svf_im1.field.data
                vel_bch2.field.data += 0.5 * cls.lie_bracket(svf_im0, svf_im1).field.data
                vel_bch2.field.data += (1 / 12.0) * cls.lie_bracket(svf_im0, cls.lie_bracket(svf_im0, svf_im1)).field

                def_result = vel_bch2.exponential()

            elif kind == 'bch2':
                vel_bch2 = copy.deepcopy(svf_im0)
                vel_bch2.field.data += svf_im1.field.data
                vel_bch2.field.data += 0.5 * cls.lie_bracket(svf_im0, svf_im1).field.data
                vel_bch2.field.data += (1 / 12.0) * (cls.lie_bracket(svf_im0, cls.lie_bracket(svf_im0, svf_im1)).field +
                                                     cls.lie_bracket(svf_im1,
                                                                       cls.lie_bracket(svf_im1, svf_im0)).field.data)
                def_result = vel_bch2.exponential()

            elif kind == 'pt':
                tmp_vel = copy.deepcopy(svf_im0)
                tmp_vel.field.data /= 2

                tmp_def_a = tmp_vel.exponential()
                tmp_vel.field.data = -tmp_vel.field.data
                tmp_def_b = tmp_vel.exponential()
                tmp_def_c = SDISP.composition(SDISP.composition(tmp_def_a, svf_im1.exponential()), tmp_def_b)

                vel_pt = copy.deepcopy(svf_im0)
                vel_pt.field.data += tmp_def_c.data
                def_result = vel_pt.exponential()

            elif kind == 'pt_alternate':

                tmp_vel = copy.deepcopy(svf_im0)
                tmp_vel.field.data /= 2

                tmp_def_a = tmp_vel.exponential()
                tmp_vel.field.data = -tmp_vel.field.data
                tmp_def_b = tmp_vel.exponentiatial()
                tmp_def_c = SDISP.composition(SDISP.composition(tmp_def_a, svf_im1.exponential()), tmp_def_b)

                vel_pt = copy.deepcopy(svf_im0)
                vel_pt.field.data += tmp_def_c.data
                def_result = vel_pt.exponential()

            else:
                raise TypeError(str_error_kind)

        elif len(kind) == 2:

            if kind[0] == 'euler' and kind[1] == 'inv_ss':
                def_im0 = svf_im0.exponential(kind=kind[0])
                def_im1 = svf_im1.exponential(kind=kind[0])
                if answer == 'disp':
                    def_result = SDISP.composition(def_im0, def_im1)
                elif answer == 'svf':
                    def_result = SDISP.composition(def_im0, def_im1).logarithm(kind=kind[1])
                else:
                    raise TypeError(str_error_answer)
            else:
                raise TypeError(str_error_kind)

        else:
            raise TypeError(str_error_kind)

        return def_result


'''
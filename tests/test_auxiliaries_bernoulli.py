from VECtorsToolkit.tools.auxiliary.bernoulli import bern, bernoulli_numb_via_poly, bernoulli_poly


''' test bernoulli number and poly '''


def test_pure_bern():
    tester = []
    flag = True
    i = 0
    while i < len(tester) and flag:
        i += 1
        if tester[i] != float(bern(i)):
            flag = False
    assert flag


def test_compare_bernoulli_poly_and_bern():
    flag = True
    i = 0
    while i < 50 and flag:
        i += 1
        if bernoulli_poly(0, i) != float(bern(i)):
            flag = False
    assert flag


def test_compare_bernoulli_numb_via_poly_and_bern():
    flag = True
    i = 0
    while i < 50 and flag:
        i += 1
        if bernoulli_numb_via_poly(i) != float(bern(i)):
            flag = False
    assert flag


if __name__ == '__main__':

    test_pure_bern()
    test_compare_bernoulli_poly_and_bern()
    test_compare_bernoulli_numb_via_poly_and_bern()

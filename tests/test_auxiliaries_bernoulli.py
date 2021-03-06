from calie.aux import bernoulli as bern

''' test bernoulli number and poly '''


def test_compare_fact():
    assert bern.fact(0) == 1
    assert bern.fact(1) == 1
    assert bern.fact(2) == 2
    assert bern.fact(3) == 6
    assert bern.fact(4) == 24


''' test bernoulli number and poly '''


def test_pure_bern():
    tester = []
    flag = True
    i = 0
    while i < len(tester) and flag:
        i += 1
        if tester[i] != float(bern.bern(i)):
            flag = False
    assert flag


def test_compare_bernoulli_poly_and_bern():
    flag = True
    i = 0
    while i < 50 and flag:
        i += 1
        if bern.bernoulli_poly(0, i) != float(bern.bern(i)):
            flag = False
    assert flag


def test_compare_bernoulli_numb_via_poly_and_bern():
    flag = True
    i = 0
    while i < 50 and flag:
        i += 1
        if bern.bernoulli_numb_via_poly(i) != float(bern.bern(i)):
            flag = False
    assert flag


if __name__ == '__main__':
    test_compare_fact()
    
    test_pure_bern()
    test_compare_bernoulli_poly_and_bern()
    test_compare_bernoulli_numb_via_poly_and_bern()

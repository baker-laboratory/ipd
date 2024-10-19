import pytest
import willutil as wu

@pytest.mark.fast
def test_movietools2():
    wu.viz.pymol_movie_script2()

@pytest.mark.fast
def test_movietools3():
    wu.viz.pymol_movie_script_objs()

def main():
    test_movietools2()

if __name__ == "__main__":
    main()

def main():
    # frames = np.array((1, 12))
    # print(frames)
    # frames[0] = [1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0]
    # cc = (0, 0, 0)

    # parse_sgdat()
    pass
    assert 0

def _highindex():
    m = set(re.findall(r"\[\d+\]", s))  # type: ignore
    m = [int(x[1:-1]) for x in m]
    print(max(m))

def parse_sgdat():
    with open("/home/sheffler/src/ipd/ipd/sym/spacegroups.txt") as inp:
        s = inp.read()
    # s = re.sub('','',s)
    s = s.replace("\n}\n", "\n   return frames, cc\n")
    s = s.replace("}\n", "")
    s = s.replace("void get_symmops_", "def symframes_")
    s = s.replace(" utility::vector1<core::kinematics::RT> &rt_out, CheshireCell &cc ) {", "):")
    s = s.replace("int ii=1", "int ii=0")
    for i in range(48):
        s = s.replace(f"[{i+1}]", f"[{i}]")
        s = s.replace(f"ii<={i+1}", f"ii<{i+1}")
    for i in range(192):
        s = s.replace(f"rt_out.resize({i+1});", f"frames = np.ones(({i+1}, 12))*12345")
    s = s.replace("rt_out[", "frames[")
    s = s.replace("core::kinematics::RT( numeric::xyzMatrix<core::Real>::rows(", "(")
    s = s.replace(")  , numeric::xyzVector<core::Real>(", ",")
    s = s.replace(") )", ")")
    s = s.replace("CheshireCell( numeric::xyzVector<core::Real>( 0, 0, 0), ", "")
    s = s.replace("numeric::xyzVector<core::Real>(", "(")
    s = s.replace("i].set_translation( frames[ii].get_translation() + ", ", 9:12] += ")
    s = s.replace("ii] = frames[ii]", "i] = frames[i]")
    s = s.replace("for ( int ii=0; ii<", "for i in range(")
    s = s.replace("; ++ii ) {", "):")
    s = s.replace(";", "")
    print(s)
    with open("spacegroup_frames.py", "w") as out:
        out.write(s)

if __name__ == "__main__":
    main()

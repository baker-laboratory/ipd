import sys
import os

# def pmlmovie1():


def pymol_movie_script_objs():
    nobjs = 203  # * 2
    movdir = "/home/sheffler/Documents/gm230203/helix_slide"
    w, h = 960, 720
    nloop = 4
    yturn = 0  # 50 / nobjs
    ray = 0

    os.makedirs(movdir, exist_ok=True)
    os.system(f"rm -f {movdir}/*.png")

    with open(f"{movdir}.pml", "w") as out:
        for i in range(nobjs):
            # for i in range(nobjs // 2):
            out.write(f"turn y, {yturn}\n")
            out.write("move_down()\n")
            out.write(f"png {movdir}/frame{i:06}.png, {w}, {h}, ray={ray}\n")

        # for i in range(nobjs // 2, nobjs):
        #   out.write(f"turn y, {-yturn}\n")
        #   out.write(f'move_up()\n')
        #   out.write(f'png {movdir}/frame{i:06}.png, {w}, {h}, ray={ray}\n')

        for iloop in range(1, nloop):
            for srcframe in range(nobjs):
                frame = iloop * nobjs + srcframe
                out.write(f'os.system("cp {movdir}/frame{srcframe:06}.png  {movdir}/frame{frame:06}.png")\n')


def pymol_movie_script1():
    home = os.path.expanduser("~")
    nstates = 20
    nframes = 640
    frame = 0

    with open(f"{home}/tmp_movie.pml", "w") as out:
        for o in (out, sys.stdout):
            o.write("mview reset\n")
            o.write(f"mset 1x{nframes}\n")
            state = 1
            for i in range((nstates - 15) * 4):
                frame += 1
                state = i // 4 + 1
                o.write(f"showbbox('MAKESYM', radius=1, scale=0.88, state={state})\n")
                o.write(f"mview store, {frame}, state={state}\n")
            for i in range(15 * 8):
                frame += 1
                state = 36 + i // 8
                o.write(f"showbbox('MAKESYM', radius=1, scale=0.88, state={state})\n")
                o.write(f"mview store, {frame}, state={state}\n")

            frame += 120
            o.write(f"mview store, {frame}, state={nstates}\n")

            for i in range(15 * 8):
                frame += 1
                state = nstates - i // 8
                o.write(f"showbbox('MAKESYM', radius=1, scale=0.88, state={state})\n")
                o.write(f"mview store, {frame}, state={state}\n")

            for i in range((nstates - 15) * 4):
                frame += 1
                state = 35 - i // 4
                o.write(f"showbbox('MAKESYM', radius=1, scale=0.88, state={state})\n")
                o.write(f"mview store, {frame}, state={state}\n")

            o.write("mplay\n")


def pymol_movie_script2():
    home = os.path.expanduser("~")
    nstates = 20

    w, h = 960, 720
    nrep1 = 5
    nrep2 = 12
    switchframe = 7
    nloop = 4
    yturn = 0  # 360 / 368
    ray = 1
    full = False

    if full:
        movdir = "/home/sheffler/Documents/gm230203/I32_mov_full"
    else:
        movdir = "/home/sheffler/Documents/gm230203/I32_mov_asym"
    os.makedirs(movdir, exist_ok=True)

    with open(f"{movdir}.pml", "w") as out:
        for o in (out,):
            frame = 0
            state = 1
            o.write("import os\n")
            o.write(f"os.system('rm -f {movdir}/*.png')\n")
            # if not full: o.write('mysetview(-Vec(1,1,1),-Vec(1,1,0))\n')
            for i in range((nstates - switchframe) * nrep1):
                state = i // nrep1 + 1
                o.write(f"frame {state}; turn y, {yturn};\n")
                # if full: o.write(f"showbbox('MAKESYM', radius=1, scale=0.88, state={state});\n")
                # else: o.write(f'showp213axes(state={state});\n')
                # o.write(f'showl632axes(state={state});\n')
                o.write(f"png {movdir}/frame{frame:06}.png, {w}, {h}, ray={ray};\n")
                frame += 1

            for i in range(switchframe * nrep2):
                state = nstates - switchframe + 1 + i // nrep2
                o.write(f"frame {state}; turn y, {yturn};\n")
                # if full: o.write(f"showbbox('MAKESYM', radius=1, scale=0.88, state={state});\n")
                # else: o.write(f'showp213axes(state={state})\n')
                # o.write(f'showl632axes(state={state});\n')
                o.write(f"png {movdir}/frame{frame:06}.png, {w}, {h}, ray={ray};\n")
                frame += 1

            for i in range(70):
                o.write(f"frame {state}; turn y, {yturn}\n")
                # if full: o.write(f"showbbox('MAKESYM', radius=1, scale=0.88, state={state});\n")
                # else: o.write(f'showp213axes(state={state});\n')
                # o.write(f'showl632axes(state={state});\n')
                o.write(f"png {movdir}/frame{frame:06}.png, {w}, {h}, ray={ray};\n")
                frame += 1

            for i in range(switchframe * nrep2):
                state = nstates - i // nrep2
                o.write(f"frame {state}; turn y, {yturn}\n")
                # if full: o.write(f"showbbox('MAKESYM', radius=1, scale=0.88, state={state});\n")
                # else: o.write(f'showp213axes(state={state});\n')
                # o.write(f'showl632axes(state={state});\n')
                o.write(f"png {movdir}/frame{frame:06}.png, {w}, {h}, ray={ray};\n")
                frame += 1

            for i in range((nstates - switchframe) * nrep1):
                state = nstates - switchframe - i // nrep1
                o.write(f"frame {state}; turn y, {yturn}\n")
                # if full: o.write(f"showbbox('MAKESYM', radius=1, scale=0.88, state={state});\n")
                # else: o.write(f'showp213axes(state={state});\n')
                # o.write(f'showl632axes(state={state});\n')
                o.write(f"png {movdir}/frame{frame:06}.png, {w}, {h}, ray={ray};\n")
                frame += 1

            nframe = frame
            ic(nframe)
            for iloop in range(1, nloop):
                for srcframe in range(nframe):
                    frame = iloop * nframe + srcframe
                    o.write(f'os.system("cp {movdir}/frame{srcframe:06}.png  {movdir}/frame{frame:06}.png")\n')

            # o.write(f'ffmpeg -framerate 24 -i {movdir}/frame%06d.png {movdir}.mp4')

def get_light_probe_path(name):
    # assert name in ['cube', 'lego']
    if name == 'cube':
        envmap_path = "./light_probe/cube.exr"
    if name == 'lego':
        envmap_path = "./light_probe/sunset.hdr"
    if name == 'cube_point':
        envmap_path = "./light_probe/point.exr"

    return envmap_path

def get_light_probe_constant(name):
    return "./light_probe/constant.exr"

def get_ply_path(name):
    # assert name in ['cube', 'lego']
    if name in ['cube', 'cube_point']:
        ply_path = "./scenes/cube_rough.obj"
    if name == 'lego':
        ply_path = "./scenes/lego.obj"

    return ply_path

def get_light_inten(name):
    # assert name in ['cube', 'lego']
    if name in ['cube']:
        inten = 5
    if name == 'lego':
        inten = 1
    if name == 'cube_point':
        inten = 20
    return inten
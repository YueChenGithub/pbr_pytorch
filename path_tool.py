def get_light_probe_path(name):
    assert name in ['cube', 'lego']
    if name == 'cube':
        envmap_path = "./light_probe/cube.exr"
    if name == 'lego':
        envmap_path = "./light_probe/sunset.hdr"

    return envmap_path

def get_ply_path(name):
    assert name in ['cube', 'lego']
    if name == 'cube':
        ply_path = "./scenes/cube_rough.obj"
    if name == 'lego':
        ply_path = "./scenes/lego.obj"

    return ply_path

def get_light_inten(name):
    assert name in ['cube', 'lego']
    if name == 'cube':
        inten = 5
    if name == 'lego':
        inten = 1
    return inten
import ffmpeg
import pathlib
import pycolmap
import open3d as o3d


def extract_frames(video_path, output_dir, fps=2):
    output_dir = pathlib.Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    (
        ffmpeg.input(str(video_path))
        .filter("fps", fps=fps)
        .output(str(output_dir / "frame_%05d.jpg"), **{"q:v": 2})
        .run(overwrite_output=True)
    )


def images_to_point_cloud(
    image_dir,
    output_dir,
    match_method="exhaustive",
    dense=False,
    use_gpu=True,
):
    image_dir = pathlib.Path(image_dir)
    output_dir = pathlib.Path(output_dir)
    database_path = output_dir / "database.db"
    mvs_path = output_dir / "mvs"
    output_dir.mkdir(parents=True, exist_ok=True)

    device = pycolmap.Device.cuda if use_gpu else pycolmap.Device.cpu

    pycolmap.extract_features(database_path, image_dir, device=device)

    if match_method == "sequential":
        pycolmap.match_sequential(database_path, device=device)
    elif match_method == "exhaustive":
        pycolmap.match_exhaustive(database_path, device=device)
    else:
        raise ValueError("match_method must be 'sequential' or 'exhaustive'")

    maps = pycolmap.incremental_mapping(database_path, image_dir, output_dir)
    if not maps:
        raise RuntimeError("Reconstruction failed — check image overlap and quality")

    reconstruction = maps[0]
    reconstruction.write(output_dir)
    reconstruction.export_PLY(output_dir / "sparse.ply")

    if dense:
        mvs_path.mkdir(exist_ok=True)
        pycolmap.undistort_images(mvs_path, output_dir, image_dir)
        pycolmap.patch_match_stereo(mvs_path)
        pycolmap.stereo_fusion(mvs_path / "dense.ply", mvs_path)

    return reconstruction


if __name__ == "__main__":
    input = "video.mp4"
    output = "input/test"
    # extract_frames(input, output)

    """
    reconstruction = images_to_point_cloud(
        image_dir="input/test",
        output_dir="output",
        match_method="sequential",
        dense=False,
    )
    """

    pcd = o3d.io.read_point_cloud("output/sparse.ply")
    o3d.visualization.draw_geometries([pcd])

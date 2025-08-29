import os
import pathlib
import zipfile
import tempfile
import typing

from fps_interpolation.timelens.timelens.common import image_sequence

'''
Read from a dataset folder. Return list of all image sequence paths.
'''
def read_dataset_directory(archive: pathlib.Path, filter=None) -> typing.List[str]:
    """
    Read folder names from an uncompressed dataset directory.
    
    Args:
        dataset_path: Path to the top-level dataset directory
        filter: String to filter folder names (optional)
    
    Returns:
        List of folder paths (relative to dataset root)
    """
    if not archive.is_dir():
        raise ValueError(f"Path {archive} is not a directory")
    
    folders = []
    
    # Walk through all directories recursively
    for root, dirs, files in os.walk(archive):
        for dir_name in dirs:
            full_dir_path = pathlib.Path(root) / dir_name
            rel_dir_path = str(full_dir_path.relative_to(archive))
            if filter is None or filter in rel_dir_path:
                folders.append(rel_dir_path)
    
    return sorted(folders)
'''
Given a video path inside the zipped folder, selectively extract it for later processing. 
Return path to uncompressed video.
'''
def extract_compressed_video(image_folder: pathlib.Path, image_file_template: str, timestamps_file: pathlib.Path):
    images = image_sequence.ImageSequence.from_folder(
        folder=image_folder,
        image_file_template=image_file_template,
        timestamps_file=timestamps_file
    )

    # Encode video as an MP4
    video_filename = image_folder.name + ".mp4"
    print(video_filename)
    #images.to_video(video_filename)

    return pathlib.Path(video_filename)

'''
Delete a temporary video file.
'''
def delete_compressed_video(vid_path: pathlib.Path):
    pass
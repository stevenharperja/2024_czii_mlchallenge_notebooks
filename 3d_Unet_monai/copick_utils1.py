import numpy as np
import zarr
import copick

class SegmentationFromPicks:
    """
    A class to handle the process of painting picks from a Copick project into a segmentation layer in Zarr format.
    
    Parameters:
    copick_config_path (str): Path to the Copick configuration JSON file.
    painting_segmentation_name (str, optional): Name of the segmentation layer to create or use. Defaults to 'paintingsegmentation'.
    """

    def __init__(self, copick_root, painting_segmentation_name=None):
        self.root = copick_root
        self.painting_segmentation_name = painting_segmentation_name or "paintingsegmentation"

    
    def get_tomogram_array(self, run, voxel_spacing=10, tomo_type="wbp"):
        _, tomogram = list(zarr.open(run.get_voxel_spacing(voxel_spacing).get_tomogram(tomo_type).zarr()).arrays())[0]
        return tomogram[:]
    
    def get_painted_segmentation_array(self, run, user_id, session_id, voxel_spacing=10, ball_radius_factor=0.5, allowlist_user_ids=None, tomo_type="wbp"):
        return self.process_run(run, user_id, session_id, voxel_spacing, ball_radius_factor, allowlist_user_ids, tomo_type)
    
    
    @staticmethod
    def create_ball(center, radius):
        """
        Create a spherical 3D array (ball) of given radius.

        Parameters:
        center (tuple): The center of the ball.
        radius (int): Radius of the ball.

        Returns:
        np.ndarray: A 3D binary array representing the ball shape.
        """
        zc, yc, xc = center
        shape = (2 * radius + 1, 2 * radius + 1, 2 * radius + 1)
        ball = np.zeros(shape, dtype=np.uint8)

        for z in range(shape[0]):
            for y in range(shape[1]):
                for x in range(shape[2]):
                    if np.linalg.norm(np.array([z, y, x]) - np.array([radius, radius, radius])) <= radius:
                        ball[z, y, x] = 1
        return ball

    def get_painting_segmentation(self, run, user_id, session_id, voxel_spacing, tomo_type):
        """
        Retrieve or create a segmentation layer for painting.

        Parameters:
        run (object): The run object to retrieve the segmentation from.
        user_id (str): The user ID performing the segmentation.
        session_id (str): The session ID used for the segmentation.
        voxel_spacing (float): The voxel spacing for scaling pick locations.
        tomo_type (str): Type of tomogram to use (e.g., denoised).

        Returns:
        tuple: (painting_seg_array, shape) where `painting_seg_array` is a Zarr dataset and `shape` is the shape of the segmentation.
        """
        segs = run.get_segmentations(
            user_id=user_id, session_id=session_id, is_multilabel=True, name=self.painting_segmentation_name, voxel_size=voxel_spacing
        )
        tomogram = run.get_voxel_spacing(voxel_spacing).get_tomogram(tomo_type)
        if not tomogram:
            return None, None
        elif len(segs) == 0:
            seg = run.new_segmentation(voxel_spacing, self.painting_segmentation_name, session_id, True, user_id=user_id)
            shape = zarr.open(tomogram.zarr(), "r")["0"].shape
            group = zarr.group(seg.path)
            group.create_dataset('data', shape=shape, dtype=np.uint16, fill_value=0)
        else:
            seg = segs[0]
            shape = zarr.open(tomogram.zarr(), "r")["0"].shape
            group = zarr.open_group(seg.path, mode="a")
            if 'data' not in group:
                group.create_dataset('data', shape=shape, dtype=np.uint16, fill_value=0)

        return group['data'], shape

    def paint_picks_as_balls(self, painting_seg_array, pick_location, segmentation_id, radius):
        """
        Paint a pick location into the segmentation array as a spherical ball.

        Parameters:
        painting_seg_array (np.ndarray): The segmentation array.
        pick_location (tuple): The (z, y, x) coordinates of the pick.
        segmentation_id (int): The ID of the segmentation label.
        radius (int): The radius of the ball to be painted.
        """
        z, y, x = pick_location
        ball = self.create_ball((radius, radius, radius), radius)

        z_min = max(0, z - radius)
        z_max = min(painting_seg_array.shape[0], z + radius + 1)
        y_min = max(0, y - radius)
        y_max = min(painting_seg_array.shape[1], y + radius + 1)
        x_min = max(0, x - radius)
        x_max = min(painting_seg_array.shape[2], x + radius + 1)

        z_ball_min = max(0, radius - z)
        z_ball_max = min(2 * radius + 1, radius + painting_seg_array.shape[0] - z)
        y_ball_min = max(0, radius - y)
        y_ball_max = min(2 * radius + 1, radius + painting_seg_array.shape[1] - y)
        x_ball_min = max(0, radius - x)
        x_ball_max = min(2 * radius + 1, radius + painting_seg_array.shape[2] - x)

        mask = ball[z_ball_min:z_ball_max, y_ball_min:y_ball_max, x_ball_min:x_ball_max] == 1
        painting_seg_array[z_min:z_max, y_min:y_max, x_min:x_max][mask] = segmentation_id

    def paint_picks(self, run, painting_seg_array, picks, segmentation_mapping, voxel_spacing, ball_radius_factor):
        """
        Paint multiple picks into the segmentation array.

        Parameters:
        run (object): The run object containing picks.
        painting_seg_array (np.ndarray): The segmentation array.
        picks (list): List of picks with object_type and location.
        segmentation_mapping (dict): Mapping from object_type to segmentation ID.
        voxel_spacing (float): Voxel spacing for pick scaling.
        ball_radius_factor (float): Factor to adjust the ball radius.
        """
        for pick in picks:
            pick_location = pick['location']
            pick_name = pick['object_type']
            segmentation_id = segmentation_mapping.get(pick_name)

            if segmentation_id is None:
                print(f"Skipping unknown object type: {pick_name}")
                continue

            z, y, x = pick_location
            z = int(z / voxel_spacing)
            y = int(y / voxel_spacing)
            x = int(x / voxel_spacing)

            particle_radius = next(obj.radius for obj in self.root.config.pickable_objects if obj.name == pick_name)
            ball_radius = int(particle_radius * ball_radius_factor / voxel_spacing)

            self.paint_picks_as_balls(painting_seg_array, (z, y, x), segmentation_id, ball_radius)

    def process_run(self, run, user_id, session_id, voxel_spacing, ball_radius_factor, allowlist_user_ids, tomo_type):
        """
        Process a single run by painting all picks into a segmentation layer.

        Parameters:
        run (object): The run to process.
        user_id (str): The user ID for segmentation.
        session_id (str): The session ID for segmentation.
        voxel_spacing (float): Voxel spacing for scaling picks.
        ball_radius_factor (float): Factor to adjust the ball radius based on pick size.
        allowlist_user_ids (list): List of allowed user IDs for segmentation.
        tomo_type (str): Type of tomogram to use for painting.
        """
        painting_seg, shape = self.get_painting_segmentation(run, user_id, session_id, voxel_spacing, tomo_type)

        if painting_seg is None:
            raise ValueError(f"Unable to obtain or create painting segmentation for run '{run.name}'.")

        segmentation_mapping = {obj.name: obj.label for obj in self.root.config.pickable_objects}
        painting_seg_array = np.zeros(shape, dtype=np.uint16)

        for obj in self.root.config.pickable_objects:
            for pick_set in run.get_picks(obj.name):
                if pick_set and pick_set.points and (not allowlist_user_ids or pick_set.user_id in allowlist_user_ids):
                    picks = [{'object_type': obj.name, 'location': (point.location.z, point.location.y, point.location.x)}
                             for point in pick_set.points]
                    self.paint_picks(run, painting_seg_array, picks, segmentation_mapping, voxel_spacing, ball_radius_factor)

        painting_seg[:] = painting_seg_array
        return painting_seg_array
    

    def process_all_runs(self, user_id, session_id, voxel_spacing, ball_radius_factor, allowlist_user_ids=None, run_name=None, tomo_type=None):
        """
        Process all runs or a specific run by painting picks into segmentation layers.

        Parameters:
        user_id (str): The user ID for segmentation.
        session_id (str): The session ID for segmentation.
        voxel_spacing (float): Voxel spacing for scaling picks.
        ball_radius_factor (float): Factor to adjust the ball radius based on pick size.
        allowlist_user_ids (list, optional): List of allowed user IDs for segmentation. Defaults to None.
        run_name (str, optional): Name of the run to process. If not provided, all runs will be processed.
        tomo_type (str, optional): Type of tomogram to use. Defaults to None.
        """
        if run_name:
            run = self.root.get_run(run_name)
            if not run:
                raise ValueError(f"Run with name '{run_name}' not found.")
            self.process_run(run, user_id, session_id, voxel_spacing, ball_radius_factor, allowlist_user_ids, tomo_type)
        else:
            for run in self.root.runs:
                self.process_run(run, user_id, session_id, voxel_spacing, ball_radius_factor, allowlist_user_ids, tomo_type)
                
                

import os
import numpy as np
import zarr
import scipy.ndimage as ndi
from skimage.segmentation import watershed
from skimage.measure import regionprops
from skimage.morphology import binary_erosion, binary_dilation, ball


class PicksFromSegmentation:
    """
    Class for processing multilabel segmentations and extracting centroids using Copick.
    """

    def __init__(self, copick_root, run_name, voxel_spacing, segmentation_idx_offset=0):
        """
        Initialize the processor with the necessary Copick root and run parameters.

        Args:
            copick_root: A CopickRootFSSpec object for managing filesystem interactions.
            run_name (str): The name of the run to process in Copick.
            voxel_spacing (int): The voxel spacing used to scale pick locations.
            segmentation_idx_offset (int): Offset applied to the segmentation indices (default 0).
        """
        self.root = copick_root
        self.run = self.root.get_run(run_name)
        self.voxel_spacing = voxel_spacing
        self.segmentation_idx_offset = segmentation_idx_offset

    def get_painting_segmentation(self, user_id, session_id, painting_segmentation_name):
        """
        Get or create a multilabel segmentation for painting. Creates a segmentation if it doesn't exist.

        Args:
            user_id (str): The ID of the user for whom the segmentation is created.
            session_id (str): The session ID.
            painting_segmentation_name (str): Name of the painting segmentation.

        Returns:
            zarr.Dataset: The segmentation dataset.
        """
        segs = self.run.get_segmentations(user_id=user_id, session_id=session_id, is_multilabel=True, name=painting_segmentation_name, voxel_size=self.voxel_spacing)
        if not self.run.get_voxel_spacing(self.voxel_spacing).get_tomogram("denoised"):
            return None
        elif len(segs) == 0:
            seg = self.run.new_segmentation(
                self.voxel_spacing, painting_segmentation_name, session_id, True, user_id=user_id
            )
            shape = zarr.open(self.run.get_voxel_spacing(self.voxel_spacing).get_tomogram("denoised").zarr(), "r")["0"].shape
            group = zarr.group(seg.path)
            group.create_dataset('data', shape=shape, dtype=np.uint16, fill_value=0)
        else:
            seg = segs[0]
            group = zarr.open_group(seg.path, mode="a")
            if 'data' not in group:
                if not self.run.get_voxel_spacing(self.voxel_spacing).get_tomogram("denoised"):
                    return None
                shape = zarr.open(self.run.get_voxel_spacing(self.voxel_spacing).get_tomogram("denoised").zarr(), "r")["0"].shape
                group.create_dataset('data', shape=shape, dtype=np.uint16, fill_value=0)
        return group['data']

    @staticmethod
    def load_multilabel_segmentation(segmentation_dir, segmentation_name, segmentation_idx_offset=0):
        """
        Load a multilabel segmentation from a Zarr file.

        Args:
            segmentation_dir (str): Directory containing the segmentation files.
            segmentation_name (str): Name of the segmentation to load.
            segmentation_idx_offset (int): Offset applied to segmentation indices (default 0).

        Returns:
            np.ndarray: The loaded segmentation array with offset applied.
        """
        segmentation_file = [f for f in os.listdir(segmentation_dir) if f.endswith('.zarr') and segmentation_name in f]
        if not segmentation_file:
            raise FileNotFoundError(f"No segmentation file found with name: {segmentation_name}")
        seg_path = os.path.join(segmentation_dir, segmentation_file[0])
        return (zarr.open(seg_path, mode='r')['data'][:] + segmentation_idx_offset)

    @staticmethod
    def detect_local_maxima(distance, maxima_filter_size=9):
        """
        Detect local maxima in the distance transform.

        Args:
            distance (np.ndarray): Distance transform of the binary mask.
            maxima_filter_size (int): Size of the maximum detection filter (default 9).

        Returns:
            np.ndarray: A binary array indicating the location of local maxima.
        """
        footprint = np.ones((maxima_filter_size, maxima_filter_size, maxima_filter_size))
        local_max = (distance == ndi.maximum_filter(distance, footprint=footprint))
        return local_max

    def get_centroids_and_save(self, segmentation, labels_to_process, user_id, session_id, min_particle_size, max_particle_size, maxima_filter_size=9):
        """
        Extract centroids from the multilabel segmentation and save them for each label.

        Args:
            segmentation (np.ndarray): Multilabel segmentation array.
            labels_to_process (list): List of labels to process.
            user_id (str): User ID for pick saving.
            session_id (str): Session ID for pick saving.
            min_particle_size (int): Minimum size threshold for particles.
            max_particle_size (int): Maximum size threshold for particles.
            maxima_filter_size (int): Size of the maximum detection filter (default 9).

        Returns:
            dict: A dictionary mapping labels to their centroids.
        """
        all_centroids = {}

        # Structuring element for erosion and dilation
        struct_elem = ball(1)  # Adjust the size of the ball as needed

        # Create a binary mask where particles are detected based on labels_to_process
        binary_mask = np.isin(segmentation, labels_to_process).astype(int)
        eroded = binary_erosion(binary_mask, struct_elem)
        dilated = binary_dilation(eroded, struct_elem)

        # Distance transform and local maxima detection
        distance = ndi.distance_transform_edt(dilated)
        local_maxi = self.detect_local_maxima(distance, maxima_filter_size=maxima_filter_size)

        # Watershed segmentation
        markers, _ = ndi.label(local_maxi)
        watershed_labels = watershed(-distance, markers, mask=dilated)

        # Compute region properties and filter based on size
        props_list = regionprops(watershed_labels)
        for region in props_list:
            label_num = region.label
            if label_num == 0:
                continue  # Skip background
            
            region_mask = watershed_labels == label_num
            original_labels_in_region = segmentation[region_mask]

            if len(original_labels_in_region) == 0:
                continue  # Skip empty regions

            unique_labels, counts = np.unique(original_labels_in_region, return_counts=True)
            dominant_label = unique_labels[np.argmax(counts)]
            
            # Use centroid of the region to assign a pick
            centroid = region.centroid
            if min_particle_size <= region.area <= max_particle_size:
                if dominant_label in all_centroids:
                    all_centroids[dominant_label].append(centroid)
                else:
                    all_centroids[dominant_label] = [centroid]

        return all_centroids

    def save_centroids_as_picks(self, all_centroids, user_id, session_id):
        """
        Save the extracted centroids as picks using Copick.

        Args:
            all_centroids (dict): Dictionary mapping labels to their centroids.
            user_id (str): User ID for pick saving.
            session_id (str): Session ID for pick saving.
        """
        for label_num, centroids in all_centroids.items():
            object_name = [obj.name for obj in self.root.pickable_objects if obj.label == label_num]
            if not object_name:
                raise ValueError(f"Label {label_num} does not correspond to any object name in pickable objects.")
            object_name = object_name[0]
            pick_set = self.run.new_picks(object_name, session_id, user_id)
            pick_set.points = [CopickPoint(location={'x': c[2] * self.voxel_spacing, 'y': c[1] * self.voxel_spacing, 'z': c[0] * self.voxel_spacing}) for c in centroids]
            pick_set.store()


def process_segmentation(copick_root, run_name, voxel_spacing, segmentation_dir, painting_segmentation_name, session_id, user_id, labels_to_process, min_particle_size=1000, max_particle_size=50000, maxima_filter_size=9, segmentation_idx_offset=0):
    """
    High-level function to process segmentation, extract centroids, and save them as picks.

    Args:
        copick_root: A CopickRootFSSpec object for managing filesystem interactions.
        run_name (str): The name of the run to process in Copick.
        voxel_spacing (int): The voxel spacing used to scale pick locations.
        segmentation_dir (str): Directory containing the multilabel segmentation.
        painting_segmentation_name (str): Name of the painting segmentation.
        session_id (str): Session ID for pick saving.
        user_id (str): User ID for pick saving.
        labels_to_process (list): List of segmentation labels to process.
        min_particle_size (int): Minimum size threshold for particles (default 1000).
        max_particle_size (int): Maximum size threshold for particles (default 50000).
        maxima_filter_size (int): Size of the maximum detection filter (default 9).
        segmentation_idx_offset (int): Offset applied to segmentation indices (default 0).
    """
    processor = PicksFromSegmentation(copick_root, run_name, voxel_spacing, segmentation_idx_offset)
    
    segmentation = processor.load_multilabel_segmentation(segmentation_dir, painting_segmentation_name, segmentation_idx_offset)
    centroids = processor.get_centroids_and_save(segmentation, labels_to_process, user_id, session_id, min_particle_size, max_particle_size, maxima_filter_size=maxima_filter_size)
    
    processor.save_centroids_as_picks(centroids, user_id, session_id)
    print("Centroid extraction and saving complete.")
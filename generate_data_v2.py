#!/home/wj17/miniconda3/envs/venv1/bin/python3.5


"""
                                                GENERATE DATA v2

* In stage 4 we concluded that there is a subtle separation between F and NF views but not enough to be a robust method for classification.
* Here, we try to clean and process the data as follows:
    * Rather than constructing the final X matrix as oriented and offsetted views, we only use offsetted views.
    * i.e. For each 2ch offsetted view, we compute the volume using the 4ch ideal view.
    * If the Simpson's volume has a percentage error > 5% (wrt to ideal simpson volume), we assign it as F (1).
    * Otherwise, as NF (0).
    * Cons: We will end up with half the number of samples than before (32000 --> 16000) <br>

* The left ventricle forms the apex of the heart. The left ventricle is thicker and more muscular than the right ventricle because it pumps blood at a higher pressure. <br>

* The right ventricle is triangular in shape and extends from the tricuspid valve in the right atrium to near the apex of the heart. <br>

* __cardiac apex, LV apex and RV apex are all different.__
* __foreshortening in our case involves LV apex.__
"""

# suppress future warnings ..
import sys, re
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=RuntimeWarning)

from classification_funcs import *
from mesh_funcs import *

"""
# ==============================================================================
# RV dir text files for HF:
# ==============================================================================
"""

# import cases ..
def vtk_str_handles(mesh_direc, type):
    os.chdir(mesh_direc)
    cases = []
    for file in glob.glob("*.vtk"):
        cases.append(file)

    num_cases = len(cases)
    print('There are ', num_cases , ' number of ', str(type), ' LV meshes.')

    return np.asarray(cases)

hc_mesh_direc = "/home/wj17/Documents/phdcoding/LV Meshes/Pablo Data 2 (MR, healthy cohort)/Circ2013 VTKmeshes/VTKmeshes/"
hf_mesh_direc = "/home/wj17/Documents/phdcoding/LV Meshes/CT Data/DATA/50_VTKmeshes/50 VTKmeshes/"

# dont use this .. not sorted to coincide with rv_dir
# hc_cases = vtk_str_handles(hc_mesh_direc, 'healthy')
# hf_cases = vtk_str_handles(hf_mesh_direc, 'heart failure')

# saving directories :
saving_direc_hc = '/home/wj17/Documents/phdcoding/src_14_nov_2018/generated_data/hc_data/'
saving_direc_hf = '/home/wj17/Documents/phdcoding/src_14_nov_2018/generated_data/hf_data/'

# import HF rv .txt files and sort:
RV_dir_path = '/home/wj17/Documents/phdcoding/LV Meshes/CT Data/DATA/Orientations3D/'

def txt_str_handles(direc):
    # gets all files that have .txt on them:
    os.chdir(direc)
    cases = []
    for file in glob.glob("*.txt"):
        cases.append(file)

    num_cases = len(cases)
    print('There are ', num_cases , ' number of text files.')

    return np.asarray(cases)

def extract_numbers(cases):
    # extracts numbers from cases which is an array of strings:
    num_cases = len(cases)
    ints = np.zeros((num_cases,), dtype = int)
    for i in range(num_cases):
        ints[i] = int(re.findall(r'\d+', cases[i])[0])

    return ints

def sort_paths(direc, typ):
    # first import strings:
    if typ == 'txt':
        cases = txt_str_handles(direc)
    else: #vtk
        cases = vtk_str_handles(direc, 'HF')

    # extract only number part:
    ints = extract_numbers(cases)

    return cases[np.argsort(ints)]

rv_cases = sort_paths(RV_dir_path, 'txt')

# sort HC and HC cases so that they coincide with order of rv_cases:
hc_cases = sort_paths(hc_mesh_direc, 'vtk')
hf_cases = sort_paths(hf_mesh_direc, 'vtk')


#  extract values.
def find_vec(substr, txtfile):
    # finds vector direction from substr line in txtfile
    with open(txtfile) as f:
        lines = f.readlines()

    for line in lines:
        if substr in line:
            rv_dir = re.findall(r"[-+]?\d*\.\d+|\d+", line)
            return rv_dir

def extract_rv_direcs(case_path, rv_cases):
    num_cases = len(rv_cases)
    # extracts all rv_direcs from rv_cases paths
    rv_dirs = np.zeros((num_cases, 3), dtype=float)
    for i in range(num_cases):
        txtfile = case_path + rv_cases[i]
        substr = 'Orientation LV to RV centre'
        rv_dir_str = np.asarray(find_vec(substr, txtfile),
                                dtype=float, order='C')
        rv_dirs[i] = rv_dir_str

    return rv_dirs

rv_dirs = extract_rv_direcs(RV_dir_path, rv_cases)

"""
# ==============================================================================
                                    main runner..
# ==============================================================================
"""

def main_runner_clean(cases, mesh_direc, binaries, rv_dirs):
    """
        Function that computes original and ideal volumes (Simpson's).

        INPUTS:
            - cases : string of vtk handles
            - mesh_direc : path to mesh directory

        OUTPUTS:
            - original volumes
            - ideal volumes based on Simpson's bi-plane method
            - landmarks_all : (2 x num_shapes x num_landmarks x dimensionality) where dimensionality is 2 (xy)
    """
    # parameters for what to run ..
    landmarks = binaries[0]
    stat_analysis = binaries[1]
    disp_segments = binaries[2]

    # filenames for segmentation ..
    fnms = []
    fnms.append(mesh_direc + "endo_labels.csv")
    fnms.append(mesh_direc + "epi_labels.csv")
    fnms.append(mesh_direc + "inner_rim_points.csv")
    fnms.append(mesh_direc + "outer_rim_points.csv")

    # variability ..
    numDegrees = 15
    lim = 30

    # foreshortening ..
    fac = 0.1 # add +10% of length of vertical lenght for each consecutive ring
    levels = 5
    pts_per_ring = [5, 5, 5, 8, 10]

    # declare outputs for Stage 1 ..
    num_cases = len(cases)
    original_volumes = np.zeros((num_cases, ), dtype=float)
    ideal_volumes = np.zeros((num_cases, ), dtype=float)

    # declare outputs for Stage 3 ..
    lvv_orient = np.zeros((len(cases), numDegrees, numDegrees), dtype=float)
    lvv_fore = np.zeros((len(cases), np.sum(pts_per_ring), np.sum(pts_per_ring)), dtype=float)

    # declare outputs for Stage 4 ..
    landmarks_all = []
    y_all = []

    # start looping through cases:
    for i, case in enumerate(cases):

        if i!=1:
            continue;

        # import vtk file ..
        vtkFile = mesh_direc + case
        print(i, " ", case)

        # create mesh instance and process it ..
        m = mesh('CT')
        m.input_mesh(vtkFile, 0)
        m.set_points_verts()
        m.align_meshes_pca(0)
        m.set_numSamples(20) #20 : number of horizontal lines
        # m.set_apex_node()
        m.set_rv_dir(rv_dirs, i)

        # extract endo epi inner outer rim data ..
        m.get_endo_epi_rim_data(fnms)

        # for display ..
        m.endoActor = m.actor_from_segment(m.endo_poly, 1, disp_segments)
        m.epiActor = m.actor_from_segment(m.epi_poly, 1, disp_segments)
        m.irActor = m.actor_from_segment(m.inner_rim_poly, 0, disp_segments)
        m.orActor = m.actor_from_segment(m.outer_rim_poly, 0, disp_segments)

        # print(m.inner_rim_poly.GetNumberOfPoints())
        # for i in range(10):
        #     print(m.endo_poly.GetFieldData().GetArray("rim labels").GetValue(i))
        #
        # sys.exit()

        # set center of inner rim poly points:
        m.set_center_irp()

        # set epi and endo apex nodes
        m.compute_apex_nodes(0)

        # get ideal view angles in order: 4ch -> 2ch and 4ch -> 3ch ..
        m.orig_view_angles = [0,-90, -150]
        m.set_original_planes(0)
        m.set_orig_cut_poly_array(0)

        for i in range(10):
            print(m.orig_cut_poly_array[0][0].GetFieldData().GetArray("rim labels").GetValue(i))

        sys.exit()

        # get originaIn clinical routine, l volume
        m.set_ideal_volume(0) # 1 = display horizontal lines
        m.compute_ground_truth_volume(0)

        ideal_volumes[i] = m.ideal_vol
        original_volumes[i] = m.ground_truth_vol

        # compute ring points
        print('ring points ..')
        ring_points = m.compute_ring_points(levels, pts_per_ring, fac, 0)

        if stat_analysis:
            print('beginning stat analysis ..')
            #  orientation test..
            fch_angle_range = np.linspace(m.orig_view_angles[0]-lim, m.orig_view_angles[0]+lim, numDegrees)
            tch_angle_range = np.linspace(m.orig_view_angles[1]-lim, m.orig_view_angles[1]+lim, numDegrees)
            lvv_orient[i] = m.orientation_test([tch_angle_range, fch_angle_range], 0)

            # foreshortening test ..
            lvv_fore[i] = m.foreshortening_test(ring_points, 0)

        if landmarks:
            print('computing landmarks for post processing ..')
            # offsetted landmarks ..
            landmarks_fore_per_mesh, labels_per_mesh = m.store_horiz_points_clean(ring_points, 1)
            # summ = np.sum(np.asarray(labels_per_mesh).flatten())
            # totall = np.asarray(labels_per_mesh).flatten().shape[0]
            # print('ratio F/NF= ', summ/float(totall))
            landmarks_all.append(landmarks_fore_per_mesh)
            y_all.append(labels_per_mesh)


    data = [lvv_orient, lvv_fore, landmarks_all, y_all]

    return original_volumes, ideal_volumes, data

"""
# ==============================================================================
#                                   RUNNER:
# ==============================================================================
"""

landmarks = 1
stat_analysis = 1
disp_segments = 0
binaries = [landmarks, stat_analysis, disp_segments]
# hc_orig_vols, hc_ideal_vols, hc_data = main_runner_clean(hc_cases, hc_mesh_direc, binaries, 0)
hf_orig_vols, hf_ideal_vols, hf_data = main_runner_clean(hf_cases, hf_mesh_direc, binaries, rv_dirs)


"""
# ==============================================================================
#                                   STORE:
# ==============================================================================
"""

store_stats = 1

def save_into_direc(array, direc, var_name):
        hkl.dump( array, direc + var_name)

if store_stats:
    # set data hc..
    hc_lvv_orient = hc_data[0]
    hc_lvv_fore = hc_data[1]
    hc_landmarks_all = hc_data[2]
    hc_y_all = hc_data[3]

    # set data hf ..
    hf_lvv_orient = hf_data[0]
    hf_lvv_fore = hf_data[1]
    hf_landmarks_all = hf_data[2]
    hf_y_all = hf_data[3]

    save_into_direc(hc_lvv_orient, saving_direc_hc, "hc_lvv_orient.hkl")
    save_into_direc(hc_lvv_fore, saving_direc_hc, "hc_lvv_fore.hkl")
    save_into_direc(hc_landmarks_all, saving_direc_hc, "hc_landmarks_all.hkl")

    save_into_direc(hf_lvv_orient, saving_direc_hf, "hf_lvv_orient.hkl" )
    save_into_direc(hf_lvv_fore, saving_direc_hf, "hf_lvv_fore.hkl")
    save_into_direc(hf_landmarks_all, saving_direc_hf, "hf_landmarks_all.hkl")

    save_into_direc(hc_y_all, saving_direc_hc, "hc_y_perr.hkl" )
    save_into_direc(hf_y_all, saving_direc_hf, "hf_y_perr.hkl" )

    save_into_direc(hc_orig_vols, saving_direc_hc, "hc_orig_vols.hkl" )
    save_into_direc(hf_orig_vols, saving_direc_hf, "hf_orig_vols.hkl" )

    save_into_direc(hc_ideal_vols, saving_direc_hc, "hc_ideal_vols.hkl" )
    save_into_direc(hf_ideal_vols, saving_direc_hf, "hf_ideal_vols.hkl" )

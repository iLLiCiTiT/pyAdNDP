import os
import sys
import re
import copy
import itertools
import collections
import warnings
import shutil

if sys.version_info[0] < 3:
    raise Exception("Must be using Python 3 version or higher!")

import numpy as np
import pickle

EXIT_WORDS = {"exit", "quit", "q"}
ADNDP_BASENAME = "AdNDP.in"
DISTANCE_BASENAME = "Distance.in"
BETA_SEP = " *******         Beta  spin orbitals         *******"


def separate_alpha_beta(nbo_path, mo_path, adndp_path, distance_path):
    alpha_dir = os.path.abspath("alpha")
    beta_dir = os.path.abspath("beta")
    if not os.path.exists(alpha_dir):
        os.mkdir(alpha_dir)

    if not os.path.exists(beta_dir):
        os.mkdir(beta_dir)

    nbo_basename = os.path.basename(nbo_path)
    mo_basename = os.path.basename(mo_path)
    adndp_basename = os.path.basename(adndp_path)
    distance_basename = os.path.basename(distance_path)

    alpha_mo_path = os.path.join(alpha_dir, mo_basename)
    beta_mo_path = os.path.join(beta_dir, mo_basename)
    alpha_nbo_path = os.path.join(alpha_dir, nbo_basename)
    beta_nbo_path = os.path.join(beta_dir, nbo_basename)
    alpha_adndp_path = os.path.join(alpha_dir, adndp_basename)
    beta_adndp_path = os.path.join(beta_dir, adndp_basename)
    alpha_distance_path = os.path.join(alpha_dir, distance_basename)
    beta_distance_path = os.path.join(beta_dir, distance_basename)

    shutil.copyfile(nbo_path, alpha_mo_path)
    shutil.copyfile(mo_path, beta_mo_path)
    shutil.copyfile(adndp_path, alpha_adndp_path)
    shutil.copyfile(adndp_path, beta_adndp_path)
    shutil.copyfile(distance_path, alpha_distance_path)
    shutil.copyfile(distance_path, beta_distance_path)

    with open(alpha_distance_path, "a") as stream:
        stream.write("\nalpha")

    with open(beta_distance_path, "a") as stream:
        stream.write("\nbeta")

    in_alpha_part = True
    with (
        open(alpha_nbo_path, "w")
    ) as a_stream, (
        open(beta_nbo_path, "w")
    ) as b_stream, (
        nbo_path
    ) as nbo_stream:
        for line in nbo_stream.readlines():
            if in_alpha_part:
                if line.startswith(BETA_SEP):
                    in_alpha_part = False
                else:
                    a_stream.write(line)
            else:
                b_stream.write(line)


class LogsReader(object):
    def __init__(self, nbo_path, mo_path):
        self._nbo_path = nbo_path
        self._mo_path = mo_path

    @property
    def nbo_path(self):
        return self._nbo_path

    @property
    def mo_path(self):
        return self._mo_path

    def readlines(self):
        with open(self.nbo_path, "r") as stream:
            for line in stream:
                yield line

    def create_adndp(self):
        # amount of atoms
        amount_of_atoms = None
        # Valence electron pairs
        valence_electron_pairs = None
        # Total electron pairs
        total_electro_pairs = None
        # Total amount of basis functions
        total_basis_funcs = None
        # Basis functions per atom
        basis_funcs_per_atom = None

        lines_queue = collections.deque(self.readlines())

        while lines_queue:
            line = lines_queue.popleft()
            if amount_of_atoms is None and "NAtoms" in line:
                amount_of_atoms = int(line.split()[1])

            if (
                valence_electron_pairs is None
                and line.startswith("   Valence")
            ):
                valence_electron_pairs = int(line[51:55]) // 2

            if total_basis_funcs is None and "basis functions," in line:
                total_basis_funcs = int(line[:7])

            if (
                total_electro_pairs is None
                and "alpha electrons"
                in line and "beta electrons" in line
            ):
                total_electro_pairs = (int(line[:7]) + int(line[24:32])) // 2

            if (
                line.startswith(
                    "   NAO  Atom  No  lang   Type(AO)    Occupancy"
                )
                or line.startswith("  NAO Atom No lang   Type(AO)")
            ):
                # Pop one line (for some reason?)
                lines_queue.popleft()
                # Basis functions per atom
                basis_funcs_per_atom = [0 for n in range(amount_of_atoms)]
                for idx in range(amount_of_atoms):
                    counter = 0
                    while len(lines_queue.popleft()) > 2:
                        counter += 1
                    basis_funcs_per_atom[idx] = counter

        basis_funcs_per_atom = basis_funcs_per_atom or []
        thresholds = [0.0 for _ in basis_funcs_per_atom]

        return AdNDPContent(
            self.nbo_path,
            self.mo_path,
            amount_of_atoms or 0,
            valence_electron_pairs or 0,
            total_electro_pairs or 0,
            total_basis_funcs or 0,
            basis_funcs_per_atom,
            thresholds
        )

    def find_residual_density(self, system, spin):
        if system == "OS":
            for line in self.readlines():
                if (
                    "alpha electrons" not in line
                    or "beta electrons" not in line
                ):
                    continue

                line_parts = line.split()
                if spin == "A":
                    return int(line_parts[0])
                return int(line_parts[3])
            return None

        alpha = None
        beta = None
        for line in self.readlines():
            if "alpha electrons" in line:
                alpha = int(line[:7])
                beta = int(line[26:32])
                if alpha > beta:
                    return alpha + beta
                break

            elif " Number of Alpha electrons" in line:
                alpha = int(line[34:])

            elif " Number of Beta electrons" in line:
                beta = int(line[34:])
                if alpha > beta:
                    return alpha + beta
                break
        return None

    def get_distince_matris(self, amount_of_atoms):
        some_calc = (
            -(-amount_of_atoms // 5)
            * (amount_of_atoms + 1)
            - 5
            * ((-(-amount_of_atoms // 5) - 1) / 2)
            * (-(-amount_of_atoms // 5)) + 1
        )

        dist = []
        dist_matrix_started = False
        dist_matrix_counter = 0
        for line in self.readlines():
            if not dist_matrix_started:
                if not line.startswith(
                    "                    Distance matrix (angstroms)"
                ):
                    continue
                dist_matrix_started = True

            dist_matrix_counter += 1
            if (
                dist_matrix_counter >= 2
                and not (line.startswith("         "))
                and dist_matrix_counter <= some_calc
            ):
                dist.append(list(map(float, line[12:].split())))
        return dist

    def get_dmnao(self):
        # TNV reading DMNAO from nbo.out
        dmnao = []
        dmnao_enabled = False
        dmnao_counter = 0
        for line in self.readlines():
            if (
                line.startswith("          NAO")
                or line.startswith("           NAO")
            ):
                dmnao_enabled = True
                dmnao_counter = 0

            if not dmnao_enabled:
                continue

            dmnao_counter += 1
            if dmnao_counter >= 3:
                if len(line) > 1:
                    dmnao.append(list(map(float, line[16:].split())))
                else:
                    dmnao_enabled = False
        return dmnao

    def get_dmao(self):
        dmao = []
        dmao_enabled = False
        dmao_counter = 0
        for line in self.readlines():
            if (
                line.startswith("          AO")
                or line.startswith("           AO")
            ):
                dmao_enabled = True
                dmao_counter = 0

            if not dmao_enabled:
                continue

            dmao_counter += 1
            if dmao_counter < 3:
                continue

            if len(line) > 1:
                if re.search("(\d)(?:\-{1})(\d)", line) is not None:
                    line = re.sub("(\d)(?:\-{1})(\d)", r"\1 -\2", line)
                dmao.append(list(map(float, line[16:].split())))
            else:
                dmao_enabled = False
        return dmao


class AdNDPContent(object):
    def __init__(
        self,
        nbo_path,
        mo_path,
        amount_of_atoms,
        valence_pairs,
        total_pairs,
        total_basis_funcs,
        basis_funcs_per_atom,
        thresholds
    ):
        self.nbo_path = nbo_path
        self.mo_path = mo_path

        self.amount_of_atoms = amount_of_atoms
        self.valence_pairs = valence_pairs
        self.total_pairs = total_pairs
        self.total_basis_funcs = total_basis_funcs
        self.thresholds = thresholds

        self.basis_funcs_per_atom = basis_funcs_per_atom

    def create_distance(self, mode, resid_save, spin):
        return DistanceContent(
            copy.deepcopy(self.thresholds), mode, resid_save, spin
        )

    def save_to_file(self, adndp_path):
        joined_bea = "\n".join(
            str(value) for value in self.basis_funcs_per_atom
        )
        joined_thresholds = "\n".join(
            str(value) for value in self.thresholds
        )
        adndp_content = (
            "NBO filename\n"
            f"{self.nbo_path}\n"
            "Number of atoms\n"
            f"{self.amount_of_atoms}\n"
            "Amount of valence electronic pairs\n"
            f"{self.valence_pairs}\n"
            "Total amount of electronic pairs\n"
            f"{self.total_pairs}\n"
            "Total amount of basis functions\n"
            f"{self.total_basis_funcs}\n"
            "Amount of basis functions on each atom\n"
            f"{joined_bea}\n"
            "Occupation number thresholds\n"
            f"{joined_thresholds}\n"
            "CMO filename\n"
            f"{self.mo_path}\n"
        )

        with open(adndp_path, "w") as stream:
            stream.write(adndp_content)

    @classmethod
    def from_file(cls, adndp_path):
        # TNV AdNDP.in reading + generating Resid.Dens.
        with open(adndp_path, "r") as stream:
            adndp_lines = stream.readlines()

        nbo_path = adndp_lines[1].strip()
        mo_path = adndp_lines[-1].strip()
        amount_of_atoms = int(adndp_lines[3])
        valence_pairs = int(adndp_lines[5])
        total_pairs = int(adndp_lines[7])
        total_basis_funcs = int(adndp_lines[9])

        basis_funcs_per_atom = []
        for idx in range(amount_of_atoms):
            basis_funcs_per_atom.append(int(adndp_lines[idx + 11]))

        thresholds = []
        for idx in range(amount_of_atoms):
            thresholds.append(float(adndp_lines[idx + amount_of_atoms + 12]))

        # TNV input warning
        if sum(basis_funcs_per_atom) != total_basis_funcs:
            print("WARNING! Number of total basis functions is wrong!")

        for threshold in thresholds:
            if threshold >= 2:
                print("WARNING! Thresholds can not be higher or equal 2!")

        return cls(
            nbo_path,
            mo_path,
            amount_of_atoms,
            valence_pairs,
            total_pairs,
            total_basis_funcs,
            basis_funcs_per_atom,
            thresholds
        )


class DistanceContent(object):
    def __init__(self, thresholds, mode, resid_save, spin):
        system = "OS"
        core_threshold = 0.99
        if spin == "0":
            system = "CS"
            core_threshold = 1.999

        self.thresholds = thresholds
        self.mode = mode
        self.resid_save = resid_save
        self.spin = spin
        self.system = system
        self.core_threshold = core_threshold

    def save_to_file(self, distance_path):
        joined_dist = " ".join(str(threshold) for threshold in self.thresholds)
        distance_content = (
            f"{joined_dist}\n"
            "Mode(LD-Late Depleting, FC-\"Found-Cut\", LDFC-hybrid): LD\n"
            "Save Residual Density Matrix: T\n"
        )
        with open(distance_path, "w+") as stream:
            stream.write(distance_content)

    @classmethod
    def from_file(cls, distance_path):
        with open(distance_path, "r") as stream:
            distance_lines = collections.deque(stream.readlines())

        thresholds = list(map(float, distance_lines.popleft().split()))
        mode = distance_lines.popleft()[54:-1]
        resid_save = distance_lines.popleft()[30:-1]
        spin = "0"
        distance_lines.popleft()
        ghost = distance_lines.popleft()
        if len(ghost) > 1:
            if ghost.startswith("alpha"):
                spin = "A"
            else:
                spin = "B"
        return cls(thresholds, mode, resid_save, spin)


def create_adndp(nbo_path, mo_path, separate, work_dir=None):
    """Create AdBDP output.

    Args:
        nbo_path (str): Path to nbo file.
        mo_path (str): Path to mo file.
        separate (bool): Separate Alpha anb Beta electron.
        work_dir (str): Where output files will be stored.
    """

    if not work_dir:
        work_dir = os.path.abspath(os.getcwd())
    nbo_path = os.path.abspath(nbo_path)
    mo_path = os.path.abspath(mo_path)

    adndp_path = os.path.join(work_dir, ADNDP_BASENAME)
    distance_path = os.path.join(work_dir, DISTANCE_BASENAME)

    reader = LogsReader(nbo_path, mo_path)

    adndp_content = reader.create_adndp()
    adndp_content.save_to_file(adndp_path)

    distance_content = adndp_content.create_distance(
        "LD", "T", "0"
    )
    distance_content.save_to_file(distance_path)

    if not separate:
        return

    print("Switching to Open Shell mode preparing mode...")

    separate_alpha_beta(
        nbo_path, mo_path, adndp_path, distance_path
    )

    print((
        "Alpha and Beta folders with proper MO, NBO,"
        f" {adndp_path} and {distance_path} files have been created."
        " To perform AdNDP analysis for openshell system, please, follow the"
        " standart procedure of AdNDP analysis using files in created folders!"
    ))


def analyse_adndp():
    #AdNDP_2.0. Tkachenko Nikolay, Boldyrev Alexander. Dec 2018.



    #Checking function
    def FileCheck(fn):
        try:
            open(fn, "r")
            return 1
        except IOError:
            return 0

    #Reading Distance.in
    if FileCheck(DISTANCE_BASENAME):
        f=open(DISTANCE_BASENAME,'r')
        Dist_thresholds=list(map(float, f.readline().split()))
        Mode=f.readline()[54:-1]
        Resid_save=f.readline()[30:-1]
        spin='0'
        f.readline()
        ghost=f.readline()
        if len(ghost)>1:
            if ghost.startswith('alpha'):
                spin='A'
            else: spin='B'

    else:
        Dist_thresholds=[0 for n in range(number_atoms)]
        Mode='LD'
        Resid_save='T'
    ####READING FILES

    #TNV AdNDP.in reading + generating Resid.Dens.
    f=open(ADNDP_BASENAME, 'r')
    file=[]
    for i in f:
        file.append(i)
    f.close()

    thresholds=[]
    basis_per_atom=[]

    NBO_file_name=file[1]
    MO_file_name=file[-1]
    number_atoms=int(file[3])
    valence_pairs=int(file[5])
    total_pairs=int(file[7])
    total_basis=int(file[9])
    Residual_density=2*total_pairs

    for i in range(number_atoms):
        basis_per_atom.append(int(file[11+i]))
    for i in range(number_atoms):
        thresholds.append(float(file[12+number_atoms+i]))
    #TNV input warning
    if sum(basis_per_atom)!=total_basis:
        print("WARNING! Number of total basis functions is wrong!")
    for i in thresholds:
        if i>=2:
            print("WARNING! Thresholds can not be higher or equal 2!")

    #In case of bugged distance.in
    if len(Dist_thresholds)<number_atoms:
        for i in range(number_atoms-len(Dist_thresholds)):
            Dist_thresholds.append(0)

    f=open(NBO_file_name[:-1], 'r')
    trig=False
    if spin=='0':
        system='CS'
    else: system='OS'

    #for i in f:
    #    if trig:
    #        if i.startswith(" Alpha Orbitals"):
    #            system='OS'
    #            break
    #        else:
    #            system='CS'
    #            break
    #    if i.startswith(" Initial guess orbital symmetries"): trig=True
    #f.close()
    if system=='OS':
        f=open(NBO_file_name[:-1], 'r')
        core_threshold=0.99
        for i in f:
            if 'alpha electrons' in i and 'beta electrons' in i:
                Div=i.split()
                alpha=int(Div[0])
                beta=int(Div[3])
                if spin=="A":
                    Residual_density=alpha
                else: Residual_density=beta
                break
        f.close()
    else:
        core_threshold=1.98
        f=open(NBO_file_name[:-1], 'r')
        for i in f:
            if "alpha electrons" in i:
                alpha=int(i[:7])
                beta=int(i[26:32])
                if alpha>beta:
                    #total_pairs-=1
                    Residual_density=alpha+beta
                break
            elif " Number of Alpha electrons" in i:
                alpha=int(i[34:])
            elif " Number of Beta electrons" in i:
                beta=int(i[34:])
                if alpha>beta:
                    #total_pairs-=1
                    Residual_density=alpha+beta
                break
        f.close()

    #Reading Distince matrix
    f=open(NBO_file_name[:-1], 'r')
    num=0
    Dist=[]
    trig=False
    for i in f:
        if i.startswith('                    Distance matrix (angstroms)') :
            trig=True
            str_counter=0
            pass
        if trig:
            str_counter+=1
            if str_counter>=2 and not(i.startswith('         ')) and str_counter<=(-(-number_atoms//5)*(number_atoms+1)-5*((-(-number_atoms//5)-1)/2)*(-(-number_atoms//5))+1):
                Dist.append(list(map(float, i[12:].split())))
        else: pass
    f.close()

    #Reshaping of distance matrix
    Dist_FOR=[]
    trig=False
    if number_atoms>=5:
        for i in range(number_atoms):
            counter=0
            while Dist[i][-1]!=0:
                counter+=1
                for j in Dist[i+number_atoms*counter-5*int(((counter+1)/2)*counter)]:
                    Dist[i].append(j)
            Dist_FOR.append(Dist[i])
    else: Dist_FOR=Dist

    #TNV reading DMNAO from nbo.out
    f=open(NBO_file_name[:-1], 'r')
    num=0
    D=[]
    trig=False
    for i in f:
        if i.startswith('          NAO') :
            trig=True
            str_counter=0
            pass
        if trig:
            str_counter+=1
            if str_counter>=3:
                if len(i)>1:
                    D.append(list(map(float, i[16:].split())))
                else:
                    trig=False
            else:
                pass
    f.close()

    #Reshaping of DMNAO
    NAO_string=total_basis
    D_FOR=D[:NAO_string].copy()
    for i in range(1,(-(-total_basis//8))):
            D_FOR=np.concatenate((D_FOR, D[i*NAO_string:(i+1)*NAO_string]),axis=1)
    #TNV
    if len(D_FOR)!=len(D_FOR[0]):
        print("WARNING! Density matrix has inproper shape!")
    ####FUNCTIONS
    #Creates indexes of block of i-th atom in DMNAO matrix
    def index_create():
        modified=[]
        Sliced_matrix=np.zeros(number_atoms**2).reshape(number_atoms,number_atoms)
        for i in range(number_atoms):
            modified.append((sum(basis_per_atom[:i]),sum(basis_per_atom[:i+1])))
        return(modified)
    indexes_D_FOR=index_create()


    #Creates all possible combinations of N centers
    def combinations(centers):
        modified=[]
        if Dist_thresholds[centers-1]!=0:
            for i in itertools.combinations(range(number_atoms), centers):
                trig=True
                for j in itertools.combinations(i, 2):
                    if Dist_FOR[j[1]][j[0]]>Dist_thresholds[centers-1]:
                        trig=False
                        break
                if trig: modified.append(i)
        else: modified=list(itertools.combinations(range(number_atoms), centers))
        return modified

    #Main function! Searching for 'centers'c-2e bonds on 'atom_centers' centers. If Resid='Y', Density matrix (D_FOR) will be changed.
    #If Core='Y', will searching for 1c-2e core orbitals with ON>1.99|e|. Returns truple (ON, wavefunction in NAO basis set)
    def bonding(centers, atom_centers, Resid='Y', Core='N'):
        dim=0
        indexes_Partition=[(0,basis_per_atom[atom_centers[0]])]
        for i in atom_centers[1:]:
            indexes_Partition.append((indexes_Partition[-1][1], indexes_Partition[-1][1]+basis_per_atom[i]))
        for i in atom_centers:
            dim+=basis_per_atom[i]
        Partition=np.zeros(dim**2).reshape(dim,dim)
        i_,j_=-1,-1
        for i in atom_centers:
            i_+=1
            j_=-1
            for j in atom_centers:
                j_+=1
                Partition[indexes_Partition[i_][0]:indexes_Partition[i_][1],indexes_Partition[j_][0]:indexes_Partition[j_][1]]=D_FOR[indexes_D_FOR[i][0]:indexes_D_FOR[i][1],indexes_D_FOR[j][0]:indexes_D_FOR[j][1]]
        ans=np.linalg.eig(Partition)
        if Core=='N' and max(ans[0])>=(2.0-thresholds[centers-1]):
            if Resid=='Y':print('FC: Occupancy of', str(centers)+'c-2e bond on ', [n+1 for n in atom_centers],'atom(s) is', np.real(max(ans[0])))
            occupancy=np.real(max(ans[0]))
            wave_function=ans[1][:,np.argmax(ans[0])]
            if Resid=='N':
                #print('_________Checking without changing of DM!___________')
                return((occupancy, wave_function))
            else:
                Partition=Partition-occupancy*wave_function*np.transpose(wave_function[np.newaxis])
                i_,j_=-1,-1
                for i in atom_centers:
                    i_+=1
                    j_=-1
                    for j in atom_centers:
                        j_+=1
                        D_FOR[indexes_D_FOR[i][0]:indexes_D_FOR[i][1],indexes_D_FOR[j][0]:indexes_D_FOR[j][1]]=Partition[indexes_Partition[i_][0]:indexes_Partition[i_][1],indexes_Partition[j_][0]:indexes_Partition[j_][1]]
                return((occupancy, wave_function))
        elif Core=='Y':
            if max(ans[0])>=core_threshold:
                print('Occupancy of Core', str(centers)+'c-2e bond on ', [n+1 for n in atom_centers],'atom(s) is', np.real(max(ans[0])))
                occupancy=np.real(max(ans[0]))
                wave_function=ans[1][:,np.argmax(ans[0])]
                Partition=Partition-occupancy*wave_function*np.transpose(wave_function[np.newaxis])
                i_,j_=-1,-1
                for i in atom_centers:
                    i_+=1
                    j_=-1
                    for j in atom_centers:
                        j_+=1
                        D_FOR[indexes_D_FOR[i][0]:indexes_D_FOR[i][1],indexes_D_FOR[j][0]:indexes_D_FOR[j][1]]=Partition[indexes_Partition[i_][0]:indexes_Partition[i_][1],indexes_Partition[j_][0]:indexes_Partition[j_][1]]
                    return((occupancy, wave_function))
            else: ans=(0,0)
        else: ans=(0,0)
        return(ans)

    #Searching for all core orbitals in molecule (will not visualized)
    def core_cut():
        nonlocal Residual_density
        core_found=0
        while core_found<(total_pairs-valence_pairs):
            for i in combinations(1):
                ans=bonding(1,i,'Y','Y')
                Residual_density=Residual_density-ans[0]
                if ans[0]!=0:
                    core_found+=1
                if core_found>=(total_pairs-valence_pairs): break

    #Searching for all valence bonds. ATTANTION! If threshold in AdNDP.in==0. than algorithm will ignore this type of bonds.
    Visual=[[] for n in range(number_atoms)] #Matrix for WF
    def bonding_Search_FC():
        nonlocal Residual_density
        nonlocal Visual
        for i in range(number_atoms):
            if thresholds[i]!=0:
                ans=[]
                if len(combinations(i+1))==0:
                        print("!!!!!!NO BONDS WITH SUCH DISTANCE RESTRICTION!!!!!!")
                for j in combinations(i+1):
                    ans.append((bonding(i+1,j,'N')[0],j))
                ans.sort(reverse=1)
                #print(ans)
                for j in ans:
                    res=(1,)
                    #print(j)
                    while res[0]!=0 and (Residual_density-j[0])>0:
                        res=bonding(i+1,j[1],'Y')
                        if res[0]!=0: Visual[i].append((j[1],np.real(res[1])))
                        Residual_density=Residual_density-res[0]
                        #print('ADDITIONAL:',j[1], res)
            else:
                print('---------------Ignoring ', str((i+1))+'c-2e bonds!---------------',)

    def dep(centers, atom_centers, occ, function):
        nonlocal D_FOR
        dim=0
        indexes_Partition=[(0,basis_per_atom[atom_centers[0]])]
        for i in atom_centers[1:]:
            indexes_Partition.append((indexes_Partition[-1][1], indexes_Partition[-1][1]+basis_per_atom[i]))
        for i in atom_centers:
            dim+=basis_per_atom[i]
        Partition=np.zeros(dim**2).reshape(dim,dim)
        i_,j_=-1,-1
        for i in atom_centers:
            i_+=1
            j_=-1
            for j in atom_centers:
                j_+=1
                Partition[indexes_Partition[i_][0]:indexes_Partition[i_][1],indexes_Partition[j_][0]:indexes_Partition[j_][1]]=D_FOR[indexes_D_FOR[i][0]:indexes_D_FOR[i][1],indexes_D_FOR[j][0]:indexes_D_FOR[j][1]]
        occupancy=occ
        wave_function=function
        Partition=Partition-occupancy*wave_function*np.transpose(wave_function[np.newaxis])
        i_,j_=-1,-1
        for i in atom_centers:
                i_+=1
                j_=-1
                for j in atom_centers:
                    j_+=1
                    D_FOR[indexes_D_FOR[i][0]:indexes_D_FOR[i][1],indexes_D_FOR[j][0]:indexes_D_FOR[j][1]]=Partition[indexes_Partition[i_][0]:indexes_Partition[i_][1],indexes_Partition[j_][0]:indexes_Partition[j_][1]]

    def bonding_Search_LDFC():
        nonlocal Residual_density
        nonlocal Visual
        for i in range(number_atoms):
            if thresholds[i]!=0:
                ans=[]
                if len(combinations(i+1))==0:
                        print("!!!!!!NO BONDS WITH SUCH DISTANCE RESTRICTION!!!!!!")
                for j in combinations(i+1):
                    ans.append((bonding(i+1,j,'N')[0],bonding(i+1,j,'N')[1],j))
                ans.sort(key=lambda tup: tup[0], reverse=True)
                for j in range(len(ans)):
                    if ans[j][0]!=0 and Residual_density-ans[j][0]>0:
                        dep(i+1, ans[j][2], ans[j][0], ans[j][1])
                        print('LDFC: Occupancy of ', str(i+1)+'c-2e bond on ', [n+1 for n in ans[j][2]], 'atom(s) is ', ans[j][0])
                        Residual_density-=ans[j][0]
                        Visual[i].append((ans[j][2],np.real(ans[j][1])))
                        res=(1,)
                        while res[0]!=0 and (Residual_density-res[0])>0:
                            #print('Hi')
                            res=bonding(i+1,ans[j][2],'Y')
                            if res[0]!=0: Visual[i].append((ans[j][2],np.real(res[1])))
                            Residual_density-=res[0]
            else:
                print('---------------Ignoring ', str((i+1))+'c-2e bonds!---------------',)

    def bonding_Search_LD():
        nonlocal Residual_density
        nonlocal Visual
        Counter=1
        for i in range(number_atoms):
            if thresholds[i]!=0:
                ans=[[1,],]
                while ans[0][0]!=0 and Residual_density-ans[0][0]>=0:
                    ans=[]
                    if len(combinations(i+1))==0:
                        print("!!!!!!NO BONDS WITH SUCH DISTANCE RESTRICTION!!!!!!")
                        break
                    for j in combinations(i+1):
                        ans.append((bonding(i+1,j,'N')[0],bonding(i+1,j,'N')[1],j))
                    ans.sort(key=lambda tup: tup[0], reverse=True)
                    for j in range(len(ans)):
                        if ans[j][0]!=0 and Residual_density-ans[j][0]>0:
                            dep(i+1, ans[j][2], ans[j][0], ans[j][1])
                            print(str(Counter)+') LD: Occupancy of ', str(i+1)+'c-2e bond on ', [n+1 for n in ans[j][2]], 'atom(s) is ', ans[j][0])
                            Counter+=1
                            Residual_density-=ans[j][0]
                            Visual[i].append((ans[j][2],np.real(ans[j][1])))
            else:
                print('---------------Ignoring ', str((i+1))+'c-2e bonds!---------------',)

    ####MAIN PROGRAMM
    core_cut()
    print('*************Residual density: ', Residual_density, '|e|*************')
    print('\n\n')
    if Mode=='LDFC':
        bonding_Search_LDFC()
    elif Mode=='FC':
        bonding_Search_FC()
    elif Mode=='LD':
        bonding_Search_LD()
    print('*************Residual density: ', Residual_density, '|e|*************')
    if Resid_save=='T':
        f=open('Resid.data','wb')
        pickle.dump((D_FOR,Residual_density),f)
        f.close

    ####VISUALISING AND CREATION NEW MO FILE
    #TNV reading basis NAO to AO
    f=open(NBO_file_name[:-1], 'r')
    num=0
    trig=False
    B=[]
    for i in f:
        if i.startswith('          AO') :
            trig=True
            str_counter=0
            pass
        if trig:
            str_counter+=1
            if str_counter>=3:
                if len(i)>1:
                    if re.search('(\d)(?:\-{1})(\d)', i)!=None:
                        new_i=re.sub('(\d)(?:\-{1})(\d)', r'\1 -\2', i)
                        B.append(list(map(float, new_i[16:].split())))
                    else:B.append(list(map(float, i[16:].split())))
                else:
                    trig=False
            else:
                pass
    f.close()
    #Reshaping of NAOAO
    B_FOR=B[:NAO_string].copy()
    Number_of_columns=len(B[0])
    Number_of_AO=(int(len(B)/total_basis)-1)*Number_of_columns+len(B[-1])
    for i in range(1,(-(-Number_of_AO//Number_of_columns))):
            B_FOR=np.concatenate((B_FOR, B[i*NAO_string:(i+1)*NAO_string]),axis=1)
    #TNV
    if len(B_FOR)!=len(B_FOR[0]):
        print("WARNING! Density matrix has inproper shape!")
    #Transform WF to AO basis set
    def Visualise(Visual):
        ans=[]
        for i in Visual:
            for j in i:
                dim=0
                indexes_Partition=[(0,basis_per_atom[j[0][0]])]
                for k in j[0][1:]:
                    indexes_Partition.append((indexes_Partition[-1][1], indexes_Partition[-1][1]+basis_per_atom[k]))
                for k in j[0]:
                    dim+=basis_per_atom[k]
                Partition_Basis=np.zeros(dim*total_basis).reshape(total_basis,dim)
                i_,j_=-1,-1
                for k in j[0]:
                    i_+=1
                    j_+=1
                    Partition_Basis[:,indexes_Partition[i_][0]:indexes_Partition[i_][1]]=B_FOR[:,indexes_D_FOR[k][0]:indexes_D_FOR[k][1]]
                ans.append(np.dot(Partition_Basis,j[1]))
        return(ans)
    VISS=Visualise(Visual)

    if len(VISS)==0: print("****Nothing to visualize!****")
    else:
        Matrix_visual=np.transpose(VISS[0][np.newaxis])
        for i in VISS[1:]:
            Matrix_visual=np.hstack((Matrix_visual, np.transpose(i[np.newaxis])))
        if np.shape(Matrix_visual)[1]%5!=0:
            for i in range(5-np.shape(Matrix_visual)[1]%5):
                Matrix_visual=np.hstack((Matrix_visual, np.transpose(np.zeros(total_basis)[np.newaxis])))


        new=open('mo_new.out', 'w')
        f=open(MO_file_name[:-1], 'r')
        num=0
        trig=False
        str_counter=0
        cycle=0
        for i in f:
            if i.startswith('     Molecular Orbital Coefficients') or i.startswith('     Alpha Molecular Orbital Coefficients'):
                trig=True
            if str_counter==(total_basis+4) and cycle!=-(-np.shape(Matrix_visual)[1]//5)-1:
                str_counter=1
                cycle+=1
            if trig:
                str_counter+=1
                if str_counter>=5 and str_counter<=(total_basis+4):
                    line = i[:23]
                    for orb in Matrix_visual[str_counter-5,cycle*5:(cycle+1)*5]:
                        if orb>=0: #chtobu vse bulo v kolonky
                            line=line+' '+'{0:.5f}'.format(np.real(orb))+'  '
                        else:
                            line=line+'{0:.5f}'.format(np.real(orb))+'  '
                    #print(line)
                    new.write(line+'\n')
                else:new.write(i)
            else:new.write(i)
        new.close()
        f.close()


def direct_search_adndp_interactive():
    f=open('Resid.data', 'rb')
    D_FOR,Residual_density=pickle.load(f)
    f.close()
    ####READING FILES
    #Checking function
    def FileCheck(fn):
        try:
            open(fn, "r")
            return 1
        except IOError:
            return 0

    #Reading Distance.in
    if FileCheck(DISTANCE_BASENAME):
        f=open(DISTANCE_BASENAME,'r')
        Dist_thresholds=list(map(float, f.readline().split()))
        Mode=f.readline()[54:-1]
        Resid_save=f.readline()[30:]
    else:
        Dist_thresholds=[0 for n in range(number_atoms)]
        Mode='LDFC'
        Resid_save='F'
    #TNV AdNDP.in reading + generating Resid.Dens.
    f=open(ADNDP_BASENAME, 'r')
    file=[]
    for i in f:
        file.append(i)
    f.close()

    thresholds=[]
    basis_per_atom=[]

    NBO_file_name=file[1]
    MO_file_name=file[-1]
    number_atoms=int(file[3])
    valence_pairs=int(file[5])
    total_pairs=int(file[7])
    total_basis=int(file[9])

    for i in range(number_atoms):
        basis_per_atom.append(int(file[11+i]))
    for i in range(number_atoms):
        thresholds.append(float(file[12+number_atoms+i]))
    #TNV input warning
    if sum(basis_per_atom)!=total_basis:
        print("WARNING! Number of total basis functions is wrong!")
    for i in thresholds:
        if i>=2:
            print("WARNING! Thresholds can not be higher or equal 2!")

    #In case of bugged distance.in
    if len(Dist_thresholds)<number_atoms:
        for i in range(number_atoms-len(Dist_thresholds)):
            Dist_thresholds.append(0)
    ####FUNCTIONS
    #Creates indexes of block of i-th atom in DMNAO matrix
    def index_create():
        modified=[]
        Sliced_matrix=np.zeros(number_atoms**2).reshape(number_atoms,number_atoms)
        for i in range(number_atoms):
            modified.append((sum(basis_per_atom[:i]),sum(basis_per_atom[:i+1])))
        return(modified)
    indexes_D_FOR=index_create()


    #Main function! Searching for 'centers'c-2e bonds on 'atom_centers' centers. If Resid='Y', Density matrix (D_FOR) will be changed.
    #If Core='Y', will searching for 1c-2e core orbitals with ON>1.99|e|. Returns truple (ON, wavefunction in NAO basis set)
    def bonding(centers, atom_centers, Resid='Y', Core='N'):
        dim=0
        indexes_Partition=[(0,basis_per_atom[atom_centers[0]])]
        for i in atom_centers[1:]:
            indexes_Partition.append((indexes_Partition[-1][1], indexes_Partition[-1][1]+basis_per_atom[i]))
        for i in atom_centers:
            dim+=basis_per_atom[i]
        Partition=np.zeros(dim**2).reshape(dim,dim)
        i_,j_=-1,-1
        for i in atom_centers:
            i_+=1
            j_=-1
            for j in atom_centers:
                j_+=1
                Partition[indexes_Partition[i_][0]:indexes_Partition[i_][1],indexes_Partition[j_][0]:indexes_Partition[j_][1]]=D_FOR[indexes_D_FOR[i][0]:indexes_D_FOR[i][1],indexes_D_FOR[j][0]:indexes_D_FOR[j][1]]
        ans=np.linalg.eig(Partition)
        if Core=='N':
            if Resid=='Y':print('Occupancy of', str(centers)+'c-2e bond on ', [n+1 for n in atom_centers],'atom(s) is', np.real(max(ans[0])))
            occupancy=np.real(max(ans[0]))
            wave_function=ans[1][:,np.argmax(ans[0])]
            if Resid=='N':
                #print('_________Checking without changing of DM!___________')
                return((occupancy, wave_function))
            else:
                Partition=Partition-occupancy*wave_function*np.transpose(wave_function[np.newaxis])
                i_,j_=-1,-1
                for i in atom_centers:
                    i_+=1
                    j_=-1
                    for j in atom_centers:
                        j_+=1
                        D_FOR[indexes_D_FOR[i][0]:indexes_D_FOR[i][1],indexes_D_FOR[j][0]:indexes_D_FOR[j][1]]=Partition[indexes_Partition[i_][0]:indexes_Partition[i_][1],indexes_Partition[j_][0]:indexes_Partition[j_][1]]
                return((occupancy, wave_function))
        else:
            if max(ans[0])>=1.99:
                print('Occupancy of Core', str(centers)+'c-2e bond on ', [n+1 for n in atom_centers],'atom(s) is', np.real(max(ans[0])))
                occupancy=np.real(max(ans[0]))
                wave_function=ans[1][:,np.argmax(ans[0])]
                Partition=Partition-occupancy*wave_function*np.transpose(wave_function[np.newaxis])
                i_,j_=-1,-1
                for i in atom_centers:
                    i_+=1
                    j_=-1
                    for j in atom_centers:
                        j_+=1
                        D_FOR[indexes_D_FOR[i][0]:indexes_D_FOR[i][1],indexes_D_FOR[j][0]:indexes_D_FOR[j][1]]=Partition[indexes_Partition[i_][0]:indexes_Partition[i_][1],indexes_Partition[j_][0]:indexes_Partition[j_][1]]
                    return((occupancy, wave_function))
            else: ans=(0,0)
        return(ans)

    def bonding_FR(centers, atom_centers, fragment):
        dim=0
        indexes_Partition=[(0,basis_per_atom[atom_centers[0]])]
        for i in atom_centers[1:]:
            indexes_Partition.append((indexes_Partition[-1][1], indexes_Partition[-1][1]+basis_per_atom[i]))
        for i in atom_centers:
            dim+=basis_per_atom[i]
        Partition=np.zeros(dim**2).reshape(dim,dim)
        i_,j_=-1,-1
        for i in atom_centers:
            i_+=1
            j_=-1
            for j in atom_centers:
                j_+=1
                Partition[indexes_Partition[i_][0]:indexes_Partition[i_][1],indexes_Partition[j_][0]:indexes_Partition[j_][1]]=D_FOR[indexes_D_FOR[i][0]:indexes_D_FOR[i][1],indexes_D_FOR[j][0]:indexes_D_FOR[j][1]]
        ans=np.linalg.eig(Partition)
        print('Direct search on Fragment ',fragment,': Occupancy of', str(centers)+'c-2e bond on ', [n+1 for n in atom_centers],'atom(s) is', np.real(max(ans[0])))
        occupancy=np.real(max(ans[0]))
        wave_function=ans[1][:,np.argmax(ans[0])]
        Partition=Partition-occupancy*wave_function*np.transpose(wave_function[np.newaxis])
        i_,j_=-1,-1
        for i in atom_centers:
            i_+=1
            j_=-1
            for j in atom_centers:
                j_+=1
                D_FOR[indexes_D_FOR[i][0]:indexes_D_FOR[i][1],indexes_D_FOR[j][0]:indexes_D_FOR[j][1]]=Partition[indexes_Partition[i_][0]:indexes_Partition[i_][1],indexes_Partition[j_][0]:indexes_Partition[j_][1]]
        return((occupancy, wave_function))

    #Searching for all valence bonds. ATTANTION! If threshold in AdNDP.in==0. than algorithm will ignore this type of bonds.
    Visual=[[] for n in range(number_atoms)] #Matrix for WF

    def bonding_Search_FR(Orbitals,add,fragment):
        nonlocal Residual_density
        nonlocal Visual
        for i in range(Orbitals):
            ans=bonding_FR(len(add),add,fragment)
            Visual[len(add)-1].append((add, ans[1]))
            Residual_density-=ans[0]

    def dep(centers, atom_centers, occ, function):
        nonlocal D_FOR
        dim=0
        indexes_Partition=[(0,basis_per_atom[atom_centers[0]])]
        for i in atom_centers[1:]:
            indexes_Partition.append((indexes_Partition[-1][1], indexes_Partition[-1][1]+basis_per_atom[i]))
        for i in atom_centers:
            dim+=basis_per_atom[i]
        Partition=np.zeros(dim**2).reshape(dim,dim)
        i_,j_=-1,-1
        for i in atom_centers:
            i_+=1
            j_=-1
            for j in atom_centers:
                j_+=1
                Partition[indexes_Partition[i_][0]:indexes_Partition[i_][1],indexes_Partition[j_][0]:indexes_Partition[j_][1]]=D_FOR[indexes_D_FOR[i][0]:indexes_D_FOR[i][1],indexes_D_FOR[j][0]:indexes_D_FOR[j][1]]
        occupancy=occ
        wave_function=function
        Partition=Partition-occupancy*wave_function*np.transpose(wave_function[np.newaxis])
        i_,j_=-1,-1
        for i in atom_centers:
                i_+=1
                j_=-1
                for j in atom_centers:
                    j_+=1
                    D_FOR[indexes_D_FOR[i][0]:indexes_D_FOR[i][1],indexes_D_FOR[j][0]:indexes_D_FOR[j][1]]=Partition[indexes_Partition[i_][0]:indexes_Partition[i_][1],indexes_Partition[j_][0]:indexes_Partition[j_][1]]

    def bonding_Search_LD_FR(add, orbitals):
        nonlocal Residual_density
        nonlocal Visual
        for i in range(Orbitals):
            ans=[]
            for j in add:
                ans.append((bonding(len(j)+1,j,'N')[0],bonding(len(j),j,'N')[1],j))
            ans.sort(key=lambda tup: tup[0], reverse=True)
            for j in range(len(ans)):
                if ans[j][0]!=0:
                    dep(i+1, ans[j][2], ans[j][0], ans[j][1])
                    print('LD: Occupancy of ', str(i+1)+'c-2e bond on ', [n+1 for n in ans[j][2]], 'atom(s) is ', ans[j][0])
                    Residual_density-=ans[j][0]
                    Visual[i].append((ans[j][2],np.real(ans[j][1])))


        ####MAIN PROGRAMM
    print('---------------Residual density: ', Residual_density, '|e|----------------\n')
    if input('Symmetry search?(Y/N): ')=="N":
        Fragments=input('Enter the number of Fragments: ')
        Fragment_list=[]
        for i in range(int(Fragments)):
            add=list(map(int,input('Enter centers for '+str(i+1)+' Fragment: ').split()))
            if add==[-1]:
                add=[n for n in range(number_atoms)]
            else:
                add=[n-1 for n in add]
            add.sort()
            Orbitals=int(input('Enter the number of orbitals in the fragment: '))
            Fragment_list.append((Orbitals, add,i))
        for i in Fragment_list:
            bonding_Search_FR(i[0],i[1],i[2])
    else:
        ####MAIN PROGRAMM symmetric search
        Fragments=input('Enter the number of symmetric Fragments: ')
        Fragment_list=[]
        for i in range(int(Fragments)):
            add=list(map(int,input('Enter centers for '+str(i+1)+' Fragment: ').split()))
            if add==[-1]:
                add=[n for n in range(number_atoms)]
            else:
                add=[n-1 for n in add]
            add.sort()
            Fragment_list.append(add)
        Orbitals=int(input('Enter the number of orbitals in each fragment: '))
        bonding_Search_LD_FR(Fragment_list,Orbitals)

    print('\n---------------Residual density: ', Residual_density, '|e|----------------')

    if input('Do you want to rewrite Resid.data?(Y/N): ')=="Y":
        f=open('Resid.data','wb')
        pickle.dump((D_FOR,Residual_density),f)
        f.close
    else: print('Returning to main menu.')
    ####VISUALISING AND CREATION NEW MO FILE
    #TNV reading basis NAO to AO
    f=open(NBO_file_name[:-1], 'r')
    num=0
    trig=False
    B=[]
    for i in f:
        if i.startswith('          AO') :
            trig=True
            str_counter=0
            pass
        if trig:
            str_counter+=1
            if str_counter>=3:
                if len(i)>1:
                    if re.search('(\d)(?:\-{1})(\d)', i)!=None:
                        new_i=re.sub('(\d)(?:\-{1})(\d)', r'\1 -\2', i)
                        B.append(list(map(float, new_i[16:].split())))
                    else:B.append(list(map(float, i[16:].split())))
                else:
                    trig=False
            else:
                pass
    f.close()
    #Reshaping of NAOAO
    NAO_string=total_basis
    B_FOR=B[:NAO_string].copy()
    Number_of_columns=len(B[0])
    Number_of_AO=(int(len(B)/total_basis)-1)*Number_of_columns+len(B[-1])
    for i in range(1,(-(-Number_of_AO//Number_of_columns))):
            B_FOR=np.concatenate((B_FOR, B[i*NAO_string:(i+1)*NAO_string]),axis=1)
    #TNV
    if len(B_FOR)!=len(B_FOR[0]):
        print("WARNING! Density matrix has inproper shape!")
    #Transform WF to AO basis set
    def Visualise(Visual):
        ans=[]
        for i in Visual:
            for j in i:
                dim=0
                indexes_Partition=[(0,basis_per_atom[j[0][0]])]
                for k in j[0][1:]:
                    indexes_Partition.append((indexes_Partition[-1][1], indexes_Partition[-1][1]+basis_per_atom[k]))
                for k in j[0]:
                    dim+=basis_per_atom[k]
                Partition_Basis=np.zeros(dim*total_basis).reshape(total_basis,dim)
                i_,j_=-1,-1
                for k in j[0]:
                    i_+=1
                    j_+=1
                    Partition_Basis[:,indexes_Partition[i_][0]:indexes_Partition[i_][1]]=B_FOR[:,indexes_D_FOR[k][0]:indexes_D_FOR[k][1]]
                ans.append(np.dot(Partition_Basis,j[1]))
        return(ans)
    VISS=Visualise(Visual)

    Matrix_visual=np.transpose(VISS[0][np.newaxis])
    for i in VISS[1:]:
        Matrix_visual=np.hstack((Matrix_visual, np.transpose(i[np.newaxis])))
    if np.shape(Matrix_visual)[1]%5!=0:
        for i in range(5-np.shape(Matrix_visual)[1]%5):
            Matrix_visual=np.hstack((Matrix_visual, np.transpose(np.zeros(total_basis)[np.newaxis])))


    new=open('mo_new_FR.out', 'w')
    f=open(MO_file_name[:-1], 'r')
    num=0
    trig=False
    str_counter=0
    cycle=0
    for i in f:
        if i.startswith('     Molecular Orbital Coefficients') or i.startswith('     Alpha Molecular Orbital Coefficients'):
            trig=True
        if str_counter==(total_basis+4) and cycle!=-(-np.shape(Matrix_visual)[1]//5)-1:
            str_counter=1
            cycle+=1
        if trig:
            str_counter+=1
            if str_counter>=5 and str_counter<=(total_basis+4):
                line = i[:23]
                for orb in Matrix_visual[str_counter-5,cycle*5:(cycle+1)*5]:
                    if orb>=0: #chtobu vse bulo v kolonky
                        line=line+' '+'{0:.5f}'.format(np.real(orb))+'  '
                    else:
                        line=line+'{0:.5f}'.format(np.real(orb))+'  '
                #print(line)
                new.write(line+'\n')
            else:new.write(i)
        else:new.write(i)
    new.close()
    f.close()


def user_get_number(message):
    while True:
        response = input(message).strip().lower()
        if response in EXIT_WORDS:
            break
        try:
            return int(response)
        except Exception:
            print(f"That is not a number: {response}")
    return None


def user_get_bool(message):
    while True:
        response = input(message).strip().lower()
        if response in EXIT_WORDS:
            break

        if response in ("y", "1", "yes", "true"):
            return True

        if response in ("n", "0", "no", "false"):
            return False

        print("Please try it again...")
    return None


def user_get_path(message):
    while True:
        response = input(message)
        if response.strip().lower() in EXIT_WORDS:
            break

        if os.path.exists(response):
            return response
        print("Entered path was not found. Please try it again...")
    return None


def user_get_enum(message, values):
    while True:
        response = input(message)
        if response.strip().lower() in EXIT_WORDS:
            break

        if response in values:
            return response
        print("Please try it again...")
    return None


def user_get_list_of_numbers(message):
    while True:
        response = input(message)
        if response.strip().lower() in EXIT_WORDS:
            break

        try:
            return list(map(int, response.split()))
        except Exception:
            print("Invalid input. Please try it again...")
    return None


def create_adndp_interactive():
    separate = user_get_bool((
        "Is the density matrix calulated separetely"
        " for Alpha and Beta electron? (Y/N): "
    )).lower()
    if separate is None:
        return

    nbo_path = user_get_path("Enter NBO file name: ")
    if nbo_path is None:
        return

    mo_path = user_get_path("Enter MO file name: ")
    if mo_path is None:
        return

    create_adndp(nbo_path, mo_path, separate)


def analysis_adndp_interactive():
    analyse_adndp()


def interactive():
    chices_msg = (
        "1) Create AdNDP.in and Distance.in files.\n"
        "2) AdNDP analysis.\n"
        "3) AdNDP direct search.\n"
        "4) Quit.\n"
    )

    done = False
    while not done:
        choice = user_get_enum(
            chices_msg, ["1", "2", "3", "4"]
        )
        if choice == "1":
            create_adndp_interactive()
        elif choice == "2":
            analysis_adndp_interactive()
        elif choice == "3":
            direct_search_adndp_interactive()
        elif choice == "4" or choice is None:
            done = True

    print((
        "\nGoodbye!"
        "\nUtah State University, 2019."
        "\nCite this work as: Physical Chemistry Chemical Physics,"
        " 2019, DOI: 10.1039/C9CP00379G\n"
    ))
    return 0


def main():
    # Silence complex warning
    warnings.filterwarnings("ignore")

    return interactive()


if __name__ == "__main__":
    sys.exit(main())

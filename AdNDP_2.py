import os
import sys
import re
import copy
import itertools
import collections
import warnings
from shutil import copyfile

if sys.version_info[0] < 3:
    raise Exception("Must be using Python 3 version or higher!")

import numpy as np
import pickle

ADNDP_BASENAME = "AdNDP.in"
DISTANCE_BASENAME = "Distance.in"



class LogsReader(object):
    def __init__(self, nbo_path, mo_path):
        self._nbo_path = nbo_path
        self._mo_path = mo_path

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


def A():
    system=input('Is the density matrix calulated separetely for Alpha and Beta electron? (Y/N): ')
    nbo_fn=input('Enter NBO file name: ')
    mo_fn=input('Enter MO file name: ')
    nbo_fn = os.path.abspath(nbo_fn)
    mo_fn = os.path.abspath(mo_fn)

    reader = LogsReader(nbo_fn, mo_fn)
    adndp_content = reader.create_adndp()
    adndp_content.save_to_file(ADNDP_BASENAME)

    distance_content = adndp_content.create_distance(
        "LD", "T", "0"
    )
    distance_content.save_to_file(DISTANCE_BASENAME)

    if system=="Y":
        print('Switching to Open Shell mode preparing mode...')

        alpha_dir = os.path.abspath("alpha")
        beta_dir = os.path.abspath("beta")
        if not os.path.exists(alpha_dir):
            os.mkdir(alpha_dir)

        if not os.path.exists(beta_dir):
            os.mkdir(beta_dir)

        nbo_basename = os.path.basename(nbo_fn)
        mo_basename = os.path.basename(mo_fn)
        alpha_mo_path = os.path.join(alpha_dir, mo_basename)
        beta_mo_path = os.path.join(beta_dir, mo_basename)
        alpha_nbo_path = os.path.join(alpha_dir, nbo_basename)
        beta_nbo_path = os.path.join(beta_dir, nbo_basename)
        alpha_adndp_path = os.path.join(alpha_dir, ADNDP_BASENAME)
        beta_adndp_path = os.path.join(beta_dir, ADNDP_BASENAME)
        alpha_distance_path = os.path.join(alpha_dir, DISTANCE_BASENAME)
        beta_distance_path = os.path.join(beta_dir, DISTANCE_BASENAME)

        copyfile(mo_fn, alpha_mo_path)
        copyfile(mo_fn, beta_mo_path)
        copyfile(ADNDP_BASENAME, alpha_adndp_path)
        copyfile(ADNDP_BASENAME, beta_adndp_path)
        copyfile(DISTANCE_BASENAME, alpha_distance_path)
        copyfile(DISTANCE_BASENAME, beta_distance_path)

        f=open(alpha_distance_path, 'a')
        f.write('\nalpha')
        f.close()
        f=open(beta_distance_path, 'a')
        f.write('\nbeta')
        f.close()
        os.remove(ADNDP_BASENAME)
        os.remove(DISTANCE_BASENAME)

        f=open(nbo_fn,'r')
        g=open(alpha_nbo_path, 'w')
        for i in f:
            if not(i.startswith(' *******         Beta  spin orbitals         *******')):
                g.write(i)
            else: break
        g.close()
        g=open(beta_nbo_path, 'w')
        f.close()
        f=open(nbo_fn,'r')
        trig=False
        for i in f:
            if not(i.startswith(' *******         Alpha spin orbitals         *******')) and not(trig):
                g.write(i)
            else: trig=True
            if trig:
                if not(i.startswith(' *******         Beta  spin orbitals         *******')):
                    pass
                else:
                    g.write(i)
                    trig=False
        print('Alpha and Beta folders with proper MO, NBO, AdNDP.in and Distance.in files have been created. To perform AdNDP analysis for openshell system, please, follow the standart procedure of AdNDP analysis using files in created folders!')
        g.close()
        f.close()
def AdNDP():
    #AdNDP_2.0. Tkachenko Nikolay, Boldyrev Alexander. Dec 2018.

    warnings.filterwarnings("ignore")

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
def AdNDP_FR():
    f=open('Resid.data', 'rb')
    D_FOR,Residual_density=pickle.load(f)
    f.close()
    ####READING FILES

    #Silence complex warning
    warnings.filterwarnings("ignore")

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


def main():
    ans=True
    trigA=True
    trigF=True
    openshell=False
    while ans:
        choice=input("""1) Create AdNDP.in and Distance.in files.
    2) AdNDP analysis.
    3) AdNDP direct search.
    4) Quit.
    """)
        if choice=="1":
            A()
        elif choice=="2":
            AdNDP()
        elif choice=="3":
            AdNDP_FR()
        elif choice=="4":
            print('Goodbye!\nUtah State University, 2019.\nCite this work as: Physical Chemistry Chemical Physics, 2019, DOI: 10.1039/C9CP00379G')
            ans=False
        else:
            print('Wrong input!')
    return 0


if __name__ == "__main__":
    sys.exit(main())

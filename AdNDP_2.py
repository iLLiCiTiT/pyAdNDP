import os
import sys
import re
import copy
import itertools
import collections
import argparse
import warnings
import shutil

if sys.version_info[0] < 3:
    raise Exception("Must be using Python 3 version or higher!")

import numpy as np
import pickle

EXIT_WORDS = {"exit", "quit", "q"}
ADNDP_BASENAME = "AdNDP.in"
DISTANCE_BASENAME = "Distance.in"
RESID_BASENAME = "Resid.data"

ALPHA_SEP = " *******         Alpha spin orbitals         *******"
BETA_SEP = " *******         Beta  spin orbitals         *******"


def matrix_recalculations(
    atom_centers,
    src_matrix,
    src_indexes,
    dst_matrix,
    dst_indexes,
    only_first=False
):
    for p_i, d_i in enumerate(atom_centers):
        d_i_start, d_i_end = src_indexes[d_i]
        p_i_start, p_i_end = dst_indexes[p_i]
        for p_j, d_j in enumerate(atom_centers):
            d_j_start, d_j_end = src_indexes[d_j]
            p_j_start, p_j_end = dst_indexes[p_j]
            dst_matrix[p_i_start:p_i_end, p_j_start:p_j_end] = (
                src_matrix[d_i_start:d_i_end, d_j_start:d_j_end]
            )

        # In some cases there should happen recalculation only on first index
        if only_first:
            break


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

        self.basis_funcs_per_atom = basis_funcs_per_atom
        self.thresholds = thresholds

        self._indexes_d_for = None
        self._log_reader = None
        self._distince_matris = None

    @property
    def log_reader(self):
        if self._log_reader is None:
            self._log_reader = LogsReader(self.nbo_path, self.mo_path)
        return self._log_reader

    def get_residual_density(self, system, spin):
        residual_density = self.log_reader.find_residual_density(system, spin)
        if residual_density is None:
            residual_density = 2 * self.total_pairs
        return residual_density

    @property
    def distince_matris(self):
        if self._distince_matris is None:
            self._distince_matris = self.log_reader.get_distince_matris(
                self.amount_of_atoms
            )
        return self._distince_matris

    @property
    def indexes_d_for(self):
        if self._indexes_d_for is None:
            self._indexes_d_for = [
                (
                    sum(self.basis_funcs_per_atom[:idx]),
                    sum(self.basis_funcs_per_atom[:idx + 1])
                )
                for idx in range(self.amount_of_atoms)
            ]
        return self._indexes_d_for

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

    def fill_thresholds_from_adndp(self, adndp_content):
        thresholds_len = len(self.thresholds)
        if thresholds_len >= adndp_content.amount_of_atoms:
            return

        for _ in range(adndp_content.amount_of_atoms - thresholds_len):
            self.thresholds.append(0)

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


class AdNDPAnalysis(object):
    def __init__(self, work_dir=None):
        if not work_dir:
            work_dir = os.getcwd()
        work_dir = os.path.abspath(work_dir)
        distance_path = os.path.join(work_dir, DISTANCE_BASENAME)
        adndp_path = os.path.join(work_dir, ADNDP_BASENAME)

        adndp_content = AdNDPContent.from_file(adndp_path)

        distance_content = DistanceContent.from_file(distance_path)
        distance_content.fill_thresholds_from_adndp(adndp_content)

        self._work_dir = work_dir
        self._adndp_content = adndp_content
        self._distance_content = distance_content

        # Attributes filled dynamically on demand
        self._residual_density = None # Change value during processing
        # Reshaped distance matrix
        self._distince_matris_mod = None
        self._dmnao = None
        self._dmnao_mod = None # Change value during processing
        self._dmnao_mod_indexes = None

        self._dmao = None
        self._dmao_mod = None # Change value during processing

        self._visual = None # Change value during processing

    def analyse(self):
        mode = self._distance_content.mode
        if mode == "LDFC":
            self._bonding_search_ldfc()
        elif mode == "FC":
            self._bonding_search_fc()
        elif mode == "LD":
            self._bonding_search_ld()

    @property
    def resid_save_from_distance(self):
        return self._distance_content.resid_save == "T"

    def store_resid_to_file(self, resid_path=None):
        if not resid_path:
            resid_path = os.path.join(self._work_dir, RESID_BASENAME)

        with open(resid_path, "wb") as stream:
            pickle.dump(
                (self.dmnao_mod, self.residual_density),
                stream
            )

    @property
    def work_dir(self):
        return self._work_dir

    @property
    def amount_of_atoms(self):
        return self._adndp_content.amount_of_atoms

    @property
    def total_basis_funcs(self):
        return self._adndp_content.total_basis_funcs

    @property
    def basis_funcs_per_atom(self):
        return self._adndp_content.basis_funcs_per_atom

    @property
    def total_pairs(self):
        return self._adndp_content.total_pairs

    @property
    def valence_pairs(self):
        return self._adndp_content.valence_pairs

    @property
    def log_reader(self):
        return self._adndp_content.log_reader

    @property
    def visual(self):
        if self._visual is None:
            self._visual = [[] for _ in range(self.amount_of_atoms)]
        return self._visual

    @property
    def distince_matris_mod(self):
        if self._distince_matris_mod is None:
            distince_matris = self.distince_matris
            amount_of_atoms = self.amount_of_atoms

            if amount_of_atoms < 5:
                modified = copy.deepcopy(distince_matris)
                self._distince_matris_mod = modified
                return modified

            modified = []
            for idx in range(amount_of_atoms):
                counter = 0
                while distince_matris[idx][-1] != 0:
                    counter += 1
                    counted_idx = (
                        idx
                        + amount_of_atoms
                        * counter
                        - 5
                        * int(((counter + 1) / 2) * counter)
                    )
                    for item in distince_matris[counted_idx]:
                        distince_matris[idx].append(item)
                modified.append(distince_matris[idx])
            self._distince_matris_mod = modified

        return self._distince_matris_mod

    @property
    def core_threshold(self):
        return self._distance_content.core_threshold

    @property
    def distince_matris(self):
        return self._adndp_content.distince_matris

    @property
    def residual_density(self):
        if self._residual_density is None:
            self._residual_density = self._adndp_content.get_residual_density(
                self._distance_content.system,
                self._distance_content.spin
            )
        return self._residual_density

    @residual_density.setter
    def set_residual_density(self, value):
        self._residual_density = value

    @property
    def dmnao(self):
        if self._dmnao is None:
            self._dmnao = self.log_reader.get_dmnao()
        return self._dmnao

    @property
    def dmnao_mod(self):
        if self._dmnao_mod is None:
            # Reshaping of DMNAO
            total_basis_funcs = self.total_basis_funcs
            dmnao = self.dmnao
            dmnao_mod = dmnao[:total_basis_funcs].copy()
            for idx in range(1, (-(-total_basis_funcs // 8))):
                dmnao_mod = np.concatenate(
                    (
                        dmnao_mod,
                        dmnao[
                            idx * total_basis_funcs:
                            (idx + 1) * total_basis_funcs
                        ]
                    ),
                    axis=1
                )
            self._dmnao_mod = dmnao_mod
        return self._dmnao_mod

    @property
    def dmnao_mod_indexes(self):
        if self._dmnao_mod_indexes is None:
            # Creates indexes of block of i-th atom in DMNAO matrix
            amount_of_atoms = self.amount_of_atoms
            basis_funcs_per_atom = self.basis_funcs_per_atom
            self._dmnao_mod_indexes = [
                (
                    sum(basis_funcs_per_atom[:idx]),
                    sum(basis_funcs_per_atom[:idx + 1])
                )
                for idx in range(amount_of_atoms)
            ]
        return self._dmnao_mod_indexes

    def dmnao_mod_has_proper_shape(self):
        dmnao_mod = self.dmnao_mod
        return len(dmnao_mod) == len(dmnao_mod[0])

    @property
    def dmao(self):
        if self._dmao is None:
            self._dmao = self.log_reader.get_dmao()
        return self._dmao

    @property
    def dmao_mod(self):
        if self._dmao_mod is None:
            # Reshaping of NAOAO
            total_basis_funcs = self.total_basis_funcs
            dmao = self.dmao
            dmao_mod = dmao[:total_basis_funcs].copy()
            columns_count = len(dmao[0])
            ao_count = (
                (int(len(dmao) / total_basis_funcs)-1)
                * columns_count + len(dmao[-1])
            )
            for idx in range(1, (-(-ao_count // columns_count))):
                dmao_mod = np.concatenate(
                    (
                        dmao_mod,
                        dmao[
                            idx * total_basis_funcs:
                            (idx + 1) * total_basis_funcs
                        ]
                    ),
                    axis=1
                )
            self._dmao_mod = dmao_mod
        return self._dmao_mod

    def dmao_mod_has_proper_shape(self):
        dmao_mod = self.dmao_mod
        return len(dmao_mod) == len(dmao_mod[0])

    def combinations(self, centers):
        """Creates all possible combinations of N centers.

        Not modifying method.
        """

        amount_of_atoms = self.amount_of_atoms
        dist_thresholds = self._distance_content.thresholds

        center_treshold = dist_thresholds[centers - 1]
        if center_treshold == 0:
            return list(
                itertools.combinations(range(amount_of_atoms), centers)
            )

        distince_matris_mod = self.distince_matris_mod
        modified = []
        for center_combination in itertools.combinations(
            range(amount_of_atoms), centers
        ):
            add_combination = True
            for pos_j, pos_i in itertools.combinations(center_combination, 2):
                if distince_matris_mod[pos_i][pos_j] > center_treshold:
                    add_combination = False
                    break

            if add_combination:
                modified.append(center_combination)

        return modified

    def _search_bonding(self, centers, atom_centers, resid="Y", core="N"):
        """Searching for 'centers'c-2e bonds on 'atom_centers' centers.

        If Resid='Y', Density matrix (D_FOR) will be changed.
        #If Core='Y', will searching for 1c-2e core orbitals with ON>1.99|e|.
        Returns truple (ON, wavefunction in NAO basis set)
        """

        basis_funcs_per_atom = self.basis_funcs_per_atom
        dmnao_mod = self.dmnao_mod
        dmnao_mod_indexes = self.dmnao_mod_indexes

        indexes_partition = []
        prev_part = 0
        dim = 0
        for atom_center in atom_centers:
            atom_value = basis_funcs_per_atom[atom_center]
            dim += atom_value
            new_part = prev_part + atom_value
            indexes_partition.append((prev_part, new_part))
            prev_part = new_part

        partition = np.zeros(dim ** 2).reshape(dim, dim)
        matrix_recalculations(
            atom_centers,
            dmnao_mod,
            dmnao_mod_indexes,
            partition,
            indexes_partition
        )

        ans = np.linalg.eig(partition)
        if core == "N":
            if resid == "Y":
                print((
                    f"Occupancy of {str(centers)}"
                    f" c-2e bond on {str([n+1 for n in atom_centers])}"
                    f" atom(s) is {np.real(max(ans[0]))}"
                ))
            occupancy = np.real(max(ans[0]))
            wave_function = ans[1][:, np.argmax(ans[0])]
            if resid == "N":
                return (occupancy, wave_function)

            partition = (
                partition
                - occupancy
                * wave_function
                * np.transpose(wave_function[np.newaxis])
            )
            matrix_recalculations(
                atom_centers,
                partition,
                indexes_partition,
                dmnao_mod,
                dmnao_mod_indexes
            )
            return (occupancy, wave_function)

        if max(ans[0]) < 1.99:
            return (0, 0)

        if not atom_centers:
            return ans

        print((
            f"Occupancy of Core {str(centers)}"
            f"c-2e bond on {str([n+1 for n in atom_centers])}"
            f"atom(s) is {np.real(max(ans[0]))}"
        ))
        occupancy = np.real(max(ans[0]))
        wave_function = ans[1][:, np.argmax(ans[0])]
        partition = (
            partition
            - occupancy
            * wave_function
            * np.transpose(wave_function[np.newaxis])
        )
        matrix_recalculations(
            atom_centers,
            partition,
            indexes_partition,
            dmnao_mod,
            dmnao_mod_indexes,
            True
        )
        return (occupancy, wave_function)

    def _analysis_bonding(self, centers, atom_centers, resid="Y", core="N"):
        """Searching for 'centers'c-2e bonds on 'atom_centers' centers.

        If resid='Y', Density matrix (D_FOR) will be changed.
        If core='Y', will searching for 1c-2e core orbitals with ON>1.99|e|.
        Returns truple (ON, wavefunction in NAO basis set).
        """

        basis_funcs_per_atom = self.basis_funcs_per_atom
        core_threshold = self.core_threshold
        thresholds = self._adndp_content.thresholds
        dmnao_mod = self.dmnao_mod
        dmnao_mod_indexes = self.dmnao_mod_indexes

        indexes_partition = []
        prev_part = 0
        dim = 0
        for atom_center in atom_centers:
            atom_value = basis_funcs_per_atom[atom_center]
            dim += atom_value
            new_part = prev_part + atom_value
            indexes_partition.append((prev_part, new_part))
            prev_part = new_part

        partition = np.zeros(dim ** 2).reshape(dim, dim)
        matrix_recalculations(
            atom_centers,
            dmnao_mod,
            dmnao_mod_indexes,
            partition,
            indexes_partition
        )

        ans = np.linalg.eig(partition)
        if not atom_centers:
            return ans

        if core == "N" and max(ans[0]) >= (2.0 - thresholds[centers - 1]):
            wave_function = ans[1][:, np.argmax(ans[0])]
            occupancy = np.real(max(ans[0]))
            if resid == "N":
                return (occupancy, wave_function)

            if resid == "Y":
                print((
                    f"FC: Occupancy of {str(centers)}"
                    f" c-2e bond on {str([n + 1 for n in atom_centers])}"
                    f" atom(s) is {np.real(max(ans[0]))}"
                ))

            partition = (
                partition
                - occupancy
                * wave_function
                * np.transpose(wave_function[np.newaxis])
            )
            matrix_recalculations(
                atom_centers,
                partition,
                indexes_partition,
                dmnao_mod,
                dmnao_mod_indexes
            )
            return (occupancy, wave_function)

        if core != "Y":
            return (0, 0)

        if max(ans[0]) < core_threshold:
            return (0, 0)

        print((
            f"Occupancy of Core {str(centers)}"
            f" c-2e bond on {str([n+1 for n in atom_centers])}"
            f" atom(s) is {np.real(max(ans[0]))}"
        ))
        occupancy = np.real(max(ans[0]))
        wave_function = ans[1][:, np.argmax(ans[0])]
        partition = (
            partition
            - occupancy
            * wave_function
            * np.transpose(wave_function[np.newaxis])
        )
        matrix_recalculations(
            atom_centers,
            partition,
            indexes_partition,
            dmnao_mod,
            dmnao_mod_indexes,
            only_first=True
        )
        return (occupancy, wave_function)

    def core_cut(self):
        """Searching for all core orbitals in molecule (will not visualized)"""

        pairs_diff = self.total_pairs - self.valence_pairs
        core_found = 0
        while core_found < pairs_diff:
            for comb in self.combinations(1):
                occupancy, _wave_function = self._analysis_bonding(
                    1, comb, "Y", "Y"
                )
                self.residual_density -= occupancy
                if occupancy != 0:
                    core_found += 1

                if core_found >= pairs_diff:
                    break

    def _bonding_search_fc(self):
        visual = self.visual
        thresholds = self._adndp_content.thresholds

        for idx in range(self.amount_of_atoms):
            if thresholds[idx] == 0:
                print((
                    "---------------"
                    f"Ignoring {str((idx + 1))}c-2e bonds!"
                    "---------------"
                ))
                continue

            combinations = self.combinations(idx + 1)
            if len(combinations) == 0:
                print("!!!!!!NO BONDS WITH SUCH DISTANCE RESTRICTION!!!!!!")

            ans = [
                (
                    self._analysis_bonding(idx + 1, combination, "N")[0],
                    combination
                )
                for combination in combinations
            ]
            ans.sort(reverse=1)

            for bonding, combination in ans:
                occupancy, wave_function = 1, None

                while occupancy != 0 and (self.residual_density - bonding) > 0:
                    occupancy, wave_function = self._analysis_bonding(
                        idx + 1, combination, "Y"
                    )
                    if occupancy != 0:
                        visual[idx].append(
                            (combination, np.real(wave_function))
                        )
                    self.residual_density -= occupancy

    def dep(self, centers, atom_centers, occupancy, wave_function):
        basis_funcs_per_atom = self.basis_funcs_per_atom

        dmnao_mod = self.dmnao_mod
        dmnao_mod_indexes = self.dmnao_mod_indexes

        indexes_partition = []
        prev_part = 0
        dim = 0
        for atom_center in atom_centers:
            atom_value = basis_funcs_per_atom[atom_center]
            dim += atom_value
            new_part = prev_part + atom_value
            indexes_partition.append((prev_part, new_part))
            prev_part = new_part

        partition = np.zeros(dim ** 2).reshape(dim, dim)
        matrix_recalculations(
            atom_centers,
            dmnao_mod,
            dmnao_mod_indexes,
            partition,
            indexes_partition,
        )

        partition = (
            partition
            - occupancy
            * wave_function
            * np.transpose(wave_function[np.newaxis])
        )
        matrix_recalculations(
            atom_centers,
            partition,
            indexes_partition,
            dmnao_mod,
            dmnao_mod_indexes,
        )

    def _bonding_search_ldfc(self):
        amount_of_atoms = self.amount_of_atoms
        thresholds = self._adndp_content.thresholds
        visual = self.visual

        for atom_idx in range(amount_of_atoms):
            if thresholds[atom_idx] == 0:
                print((
                    "---------------"
                    f"Ignoring {str((atom_idx + 1))} c-2e bonds!"
                    "---------------"
                ))
                continue

            ans = []
            combinations = self.combinations(atom_idx + 1)
            if len(combinations) == 0:
                print("!!!!!!NO BONDS WITH SUCH DISTANCE RESTRICTION!!!!!!")

            for combination in combinations:
                occupancy, wave_function = self._analysis_bonding(
                    atom_idx + 1, combination, "N"
                )
                ans.append(
                    (occupancy, wave_function, combination)
                )

            ans.sort(key=lambda item: item[0], reverse=True)

            for an in ans:
                occupancy, wave_function, comb = an
                if occupancy != 0 and self.residual_density - occupancy > 0:
                    self.dep(atom_idx + 1, comb, occupancy, wave_function)
                    print((
                        f"LDFC: Occupancy of {str(atom_idx + 1)}"
                        f" c-2e bond on {str([n + 1 for n in comb])}"
                        f"atom(s) is {occupancy}"
                    ))
                    self.residual_density -= occupancy
                    visual[atom_idx].append((comb, np.real(wave_function)))

                    occupancy = 1
                    wave_function = None
                    while (
                        occupancy != 0
                        and (self.residual_density - occupancy) > 0
                    ):
                        occupancy, wave_function = self._analysis_bonding(
                            atom_idx + 1, comb, "Y"
                        )
                        if occupancy != 0:
                            visual[atom_idx].append(
                                (comb, np.real(wave_function))
                            )
                        self.residual_density -= occupancy

    def _bonding_search_ld(self):
        amount_of_atoms = self.amount_of_atoms
        thresholds = self._adndp_content.thresholds
        visual = self.visual

        counter = 1
        for idx in range(amount_of_atoms):
            if thresholds[idx] == 0:
                print((
                    "---------------"
                    f"Ignoring {str((idx + 1))} c-2e bonds!"
                    "---------------"
                ))
                continue

            ans = [[1, ], ]
            while ans[0][0] != 0 and self.residual_density - ans[0][0] >= 0:
                combinations = self.combinations(idx + 1)
                if len(combinations) == 0:
                    print(
                        "!!!!!!NO BONDS WITH SUCH DISTANCE RESTRICTION!!!!!!"
                    )
                    break

                ans = []
                for combination in combinations:
                    occupancy, wave_function = self._analysis_bonding(
                        idx + 1, combination, "N"
                    )
                    ans.append((occupancy, wave_function, combination))

                ans.sort(key=lambda item: item[0], reverse=True)

                for an in ans:
                    occupancy, wave_function, combination = an
                    if (
                        occupancy == 0
                        or self.residual_density - occupancy <= 0
                    ):
                        continue

                    self.dep(idx + 1, combination, occupancy, wave_function)
                    print((
                        f"{counter}) LD: Occupancy of {str(idx + 1)}"
                        f" c-2e bond on {str([n + 1 for n in combination])}"
                        f" atom(s) is {occupancy}"
                    ))
                    counter += 1
                    self.residual_density -= occupancy
                    visual[idx].append((combination, np.real(wave_function)))

    def bonding_search_ld_fr(self, fragments, orbitals):
        visual = self.visual
        for idx in range(orbitals):
            ans = []
            for fragment in fragments:
                occupancy, wave_function = self._search_bonding(
                    len(fragment) + 1, fragment, "N"
                )
                ans.append((
                    occupancy,
                    wave_function,
                    fragment
                ))
            ans.sort(key=lambda item: item[0], reverse=True)
            for an in ans:
                occupancy, wave_function, fragment = an
                if occupancy == 0:
                    continue

                self.dep(idx + 1, fragment, occupancy, wave_function)
                print((
                    f"LD: Occupancy of {str(len(fragment))}"
                    f"c-2e bond on {str([n + 1 for n in fragment])}"
                    f"atom(s) is {occupancy}"
                ))
                self.residual_density -= occupancy
                visual[idx].append((fragment, np.real(wave_function)))

    def bonding_search_fr(self, orbitals, add, fragment):
        visual = self.visual
        for _ in range(orbitals):
            ans = self._bonding_fr(len(add), add, fragment)
            visual[len(add) - 1].append((add, ans[1]))
            self.residual_density -= ans[0]

    def _bonding_fr(self, centers, atom_centers, fragment):
        basis_funcs_per_atom = self.basis_funcs_per_atom
        dmnao_mod = self.dmnao_mod
        dmnao_mod_indexes = self.dmnao_mod_indexes

        indexes_partition = []
        prev_part = 0
        dim = 0
        for atom_center in atom_centers:
            atom_value = basis_funcs_per_atom[atom_center]
            dim += atom_value
            new_part = prev_part + atom_value
            indexes_partition.append((prev_part, new_part))
            prev_part = new_part

        partition = np.zeros(dim ** 2).reshape(dim, dim)
        matrix_recalculations(
            atom_centers,
            dmnao_mod,
            dmnao_mod_indexes,
            partition,
            indexes_partition,
        )

        ans = np.linalg.eig(partition)
        print((
            f"Direct search on Fragment {fragment}: Occupancy of {centers}"
            f" c-2e bond on {str([n + 1 for n in atom_centers])}"
            f" atom(s) is {np.real(max(ans[0]))}"
        ))
        occupancy = np.real(max(ans[0]))
        wave_function = ans[1][:, np.argmax(ans[0])]
        partition = (
            partition
            - occupancy
            * wave_function
            * np.transpose(wave_function[np.newaxis])
        )
        matrix_recalculations(
            atom_centers,
            partition,
            indexes_partition,
            dmnao_mod,
            dmnao_mod_indexes,
        )
        return (occupancy, wave_function)

    def get_visualisation_content(self):
        visual = self.visual
        basis_funcs_per_atom = self.basis_funcs_per_atom
        total_basis_funcs = self.total_basis_funcs
        dmao_mod = self.dmao_mod
        # They have same indexes
        dmao_mod_indexes = self.dmnao_mod_indexes

        ans = []
        for items in visual:
            for item in items:
                comb, wave_function = item

                indexes_partition = []
                prev_part = 0
                dim = 0
                for atom_center in comb:
                    atom_value = basis_funcs_per_atom[atom_center]
                    dim += atom_value
                    new_part = prev_part + atom_value
                    indexes_partition.append((prev_part, new_part))
                    prev_part = new_part

                partition_basis = (
                    np
                    .zeros(dim * total_basis_funcs)
                    .reshape(total_basis_funcs, dim)
                )
                # Like 'matrix_recalculations'
                for p_i, d_i in enumerate(comb):
                    p_start, p_end = indexes_partition[p_i]
                    d_start, d_end = dmao_mod_indexes[d_i]
                    partition_basis[:, p_start:p_end] = (
                        dmao_mod[:, d_start:d_end]
                    )

                ans.append(np.dot(partition_basis, wave_function))
        return ans

    def visualise(self, new_mo_output_path):
        content = self.get_visualisation_content()
        if not content:
            return False

        total_basis_funcs = self.total_basis_funcs

        matrix_visual = None
        for item in content:
            if matrix_visual is None:
                matrix_visual = np.transpose(item[np.newaxis])
                continue
            matrix_visual = np.hstack(
                (matrix_visual, np.transpose(item[np.newaxis]))
            )

        second_item_shape = np.shape(matrix_visual)[1]
        if second_item_shape % 5 != 0:
            for _ in range(5 - second_item_shape % 5):
                matrix_visual = np.hstack(
                    (
                        matrix_visual,
                        np.transpose(np.zeros(total_basis_funcs)[np.newaxis])
                    )
                )

        after_molecular = False
        str_counter = 0
        cycle = 0
        with open(new_mo_output_path, "w") as stream:
            for line in self.log_reader.readlines():
                if not after_molecular and self._is_molecular_line(line):
                    after_molecular = True

                if not after_molecular:
                    stream.write(line)
                    continue

                if (
                    str_counter == (total_basis_funcs + 4)
                    and cycle != -(-np.shape(matrix_visual)[1] // 5) - 1
                ):
                    str_counter = 1
                    cycle += 1

                str_counter += 1
                if str_counter >= 5 and str_counter <= (total_basis_funcs + 4):
                    new_line = line[:23]
                    f_cycle = cycle * 5
                    for orb in (
                        matrix_visual[str_counter - 5, f_cycle:f_cycle + 5]
                    ):
                        l_start = ""
                        if orb >= 0:
                            l_start = " "

                        new_line += "{}{:.5f}  ".format(l_start, np.real(orb))

                    stream.write(new_line + "\n")
                else:
                    stream.write(line)

        return True

    def _is_molecular_line(self, line):
        return (
            line.startswith("     Molecular Orbital Coefficients")
            or line.startswith("     Alpha Molecular Orbital Coefficients")
        )


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


def analyse_adndp(work_dir=None):
    # AdNDP_2.0. Tkachenko Nikolay, Boldyrev Alexander. Dec 2018.
    analysis = AdNDPAnalysis(work_dir)

    # TNV
    if not analysis.dmnao_mod_has_proper_shape():
        print("WARNING! Density matrix has inproper shape!")

    # MAIN PROGRAMM
    print((
        "!Core orbitals will not be cut out and will"
        " be present in visualization file!"
        "\n*************Residual density: {} |e|*************\n\n"
    ).format(analysis.residual_density))

    analysis.analyse()

    print((
        "*************Residual density: {} |e|*************"
    ).format(analysis.residual_density))

    if analysis.resid_save_from_distance:
        analysis.store_resid_to_file()

    # VISUALISING AND CREATION NEW MO FILE
    if not analysis.dmao_mod_has_proper_shape():
        print("WARNING! Density matrix has inproper shape!")

    # Transform WF to AO basis set
    new_mo_output_path = os.path.join(analysis.work_dir, "mo_new.out")
    if not analysis.visualise(new_mo_output_path):
        print("****Nothing to visualize!****")


def direct_search_adndp_args(
    symetry_search,
    fragments,
    orbitals,
    overwrite_resid,
    work_dir=None
):
    analysis = AdNDPAnalysis(work_dir)
    print((
        "---------------"
        f"Residual density: {analysis.residual_density}"
        "|e|----------------\n"
    ))
    direct_search_adndp(
        symetry_search,
        fragments,
        orbitals,
        overwrite_resid,
        analysis
    )


def direct_search_adndp(
    fragments,
    orbitals,
    overwrite_resid,
    analysis
):
    if orbitals is None:
        for orbitals, add, idx in fragments:
            analysis.bonding_search_fr(orbitals, add, idx)
    else:
        analysis.bonding_search_ld_fr(fragments, orbitals)

    print((
        "\n---------------"
        f"Residual density: {analysis.residual_density}"
        "|e|----------------"
    ))

    if overwrite_resid:
        analysis.store_resid_to_file()

    # VISUALISING AND CREATION NEW MO FILE
    if not analysis.dmao_mod_has_proper_shape():
        print("WARNING! Density matrix has inproper shape!")

    new_mo_output_path = os.path.join(analysis.work_dir, "mo_new_FR.out")
    if not analysis.visualise(new_mo_output_path):
        print("****Nothing to visualize!****")


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


def analyse_adndp_interactive():
    analyse_adndp()


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


def direct_search_adndp_interactive():
    analysis = AdNDPAnalysis()
    print((
        "---------------"
        f"Residual density: {analysis.residual_density}"
        "|e|----------------\n"
    ))
    symetry_search = user_get_bool("Symmetry search?(Y/N): ")
    if symetry_search is None:
        return

    if symetry_search:
        count_question = "Enter the number of symmetric Fragments: "
    else:
        count_question = "Enter the number of Fragments: "
    frament_count = user_get_number(count_question)
    if frament_count is None:
        return

    fragments = []
    if not symetry_search:
        for idx in range(frament_count):
            add = user_get_list_of_numbers(
                f"Enter centers for {idx + 1} Fragment: "
            )
            if add is None:
                return

            if add == [-1]:
                add = [n for n in range(analysis.amount_of_atoms)]
            else:
                add = [n - 1 for n in add]
            add.sort()
            orbitals = user_get_number(
                "Enter the number of orbitals in the fragment: "
            )
            if orbitals is None:
                return
            fragments.append((orbitals, add, idx))

        orbitals = None

    else:
        for idx in range(int(frament_count)):
            add = user_get_list_of_numbers(
                f"Enter centers for {idx + 1} Fragment: "
            )
            if add is None:
                return
            if add == [-1]:
                add = [n for n in range(analysis.amount_of_atoms)]
            else:
                add = [n - 1 for n in add]
            add.sort()
            fragments.append(add)

        orbitals = user_get_number(
            "Enter the number of orbitals in each fragment: "
        )
        if orbitals is None:
            return

    overwrite_resid = user_get_bool(
        "Do you want to rewrite Resid.data?(Y/N): "
    )
    direct_search_adndp(
        fragments,
        orbitals,
        overwrite_resid,
        analysis
    )


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
            analyse_adndp_interactive()
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
    warnings.filterwarnings("ignore")

    main_parser = argparse.ArgumentParser()
    commands_subparser = main_parser.add_subparsers(
        title="command",
        dest="command"
    )
    create_parser = commands_subparser.add_parser(
        "create",
        help="Create AdNDP"
    )
    create_parser.add_argument(
        "--nbo_input",
        "-nbo",
        required=True,
        dest="nbo_path",
        help="Path to NBO file."
    )
    create_parser.add_argument(
        "--mo_input",
        "-mo",
        required=True,
        dest="mo_path",
        help="Path to MO file."
    )
    create_parser.add_argument(
        "--separate",
        "-s",
        default=False,
        required=False,
        action="store_true",
        dest="separate",
        help=(
            "The density matrix is calulated separetely"
            " for Alpha and Beta electron."
        )
    )
    create_parser.add_argument(
        "--workdir",
        "-w",
        required=False,
        default=None,
        dest="work_dir",
        help="Work directory where work files will be created."
    )

    analysis_parser = commands_subparser.add_parser(
        "analyse",
        help="Analysis of AdNDP"
    )
    analysis_parser.add_argument(
        "--workdir",
        "-w",
        required=False,
        default=None,
        dest="work_dir",
        help="Work directory from where work files are loaded."
    )

    search_parser = commands_subparser.add_parser(
        "search",
        help="Direct search (NOT IMPLEMENTED)"
    )

    args = main_parser.parse_args()
    command = args.command
    if command is None:
        return interactive()

    if command == "create":
        create_adndp(
            args.nbo_path, args.mo_path, args.separate, args.work_dir
        )
        return 0

    if command == "analyse":
        analyse_adndp(args.work_dir)
        return 0

    if command == "search":
        raise NotImplementedError(
            "Direct search command is not implemented yet"
        )

    raise NotImplementedError("Unknown launch arguments {}".format(
        " ".join(sys.argv)
    ))


if __name__ == "__main__":
    sys.exit(main())

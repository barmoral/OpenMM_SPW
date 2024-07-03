
# ==============================================================================
# Atom selector tool
# ==============================================================================

class _AtomSelector(object):
    """
    Helper class to select atoms from semi-arbitrary selections based on a topography
    """

    def __init__(self, topography):
        self.topography = topography

    def compute_atom_intersect(self, input_atoms, topography_key: str, *additional_sets: Iterable[int]) -> Set[int]:

        """
        Compute the intersect of atoms formed from a given input, topography key to reference, and
        any additional sets for cross reference

        Parameters
        ----------
        input_atoms : str, iterable of int, or None
            Atom-like selection which can accept a :func:`yank.Topography.select`, or a sequence of ints, or None
            When given None, only the Topography key and additional sets are used
        topography_key : str
            Key in the :class:`yank.Topography` which is used to initially sub-select atoms
        additional_sets : set of int or iterable of int
            Any additional sets to cross reference

        Returns
        -------
        atom_intersect : set
            Set of atoms intersecting the input, the topography key, and the additional sets
        """

        topography = self.topography
        topography_set = set(getattr(topography, topography_key))
        # Ensure additions are sets
        additional_sets = [set(additional_set) for additional_set in additional_sets]
        if len(additional_sets) == 0:
            # Case no sets were provided
            additional_intersect = topography_set
        else:
            additional_intersect = set.intersection(*additional_sets)

        @functools.singledispatch
        def compute_atom_set(passed_atoms):
            """Helper function for doing set operations on heavy ligand atoms of all other types"""
            input_set = set(passed_atoms)
            intersect_set = input_set & additional_intersect & topography_set
            if intersect_set != input_set:
                return intersect_set
            else:
                # This ensures if no changes are made to the set, then passed atoms are returned unmodied
                return passed_atoms

        @compute_atom_set.register(type(None))
        def compute_atom_none(_):
            """Helper for None type parsing"""
            return topography_set & additional_intersect

        @compute_atom_set.register(str)
        def compute_atom_str(input_string):
            """Helper for string parsing"""
            output = topography.select(input_string, as_set=False)  # Preserve order
            set_output = set(output)
            # Ensure the selection is in the correct set
            set_combined = set_output & topography_set & additional_intersect
            final_output = [particle for particle in output if particle in set_combined]
            # Force output to be a normal int, don't need to worry about floats at this point, there should not be any
            # If they come out as np.int64's, OpenMM complains
            return [*map(int, final_output)]

        return compute_atom_set(input_atoms)
    
# ==============================================================================
# Harmonic protein-ligand restraint.
# ==============================================================================

class Harmonic(RadiallySymmetricRestraint):
    """Impose a single harmonic restraint between ligand and protein.

    This can be used to prevent the ligand from drifting too far from the
    protein in implicit solvent calculations or to keep the ligand close
    to the binding pocket in the decoupled states to increase mixing.

    The restraint is applied between the centroids of two groups of atoms
    that belong to the receptor and the ligand respectively. The centroids
    are determined by a mass-weighted average of the group particles positions.

    The energy expression of the restraint is given by

       ``E = lambda_restraints * (K/2)*r^2``

    where `K` is the spring constant, `r` is the distance between the
    two group centroids, and `lambda_restraints` is a scale factor that
    can be used to control the strength of the restraint. You can control
    ``lambda_restraints`` through :class:`RestraintState` class.

    The class supports automatic determination of the parameters left undefined or defined by strings
    in the constructor through :func:`determine_missing_parameters`.

    With OpenCL, groups with more than 1 atom are supported only on 64bit
    platforms.

    Parameters
    ----------
    spring_constant : simtk.unit.Quantity, optional
        The spring constant K (see energy expression above) in units compatible
        with joule/nanometer**2/mole (default is None).
    restrained_receptor_atoms : iterable of int, int, or str, optional
        The indices of the receptor atoms to restrain, an MDTraj DSL expression, or a
        :class:`Topography <yank.Topography>` region name,
        or :func:`Topography Select String <yank.Topography.select>`.
        This can temporarily be left undefined, but ``determine_missing_parameters()``
        must be called before using the Restraint object. The same if a DSL
        expression or Topography region is provided (default is None).
    restrained_ligand_atoms : iterable of int, int, or str, optional
        The indices of the ligand atoms to restrain, an MDTraj DSL expression.
        or a :class:`Topography <yank.Topography>` region name,
        or :func:`Topography Select String <yank.Topography.select>`.
        This can temporarily be left undefined, but ``determine_missing_parameters()``
        must be called before using the Restraint object. The same if a DSL
        expression or Topography region is provided (default is None).

    Attributes
    ----------
    restrained_receptor_atoms : list of int, str, or None
        The indices of the receptor atoms to restrain, an MDTraj selection string, or a Topography region selection
        string.
    restrained_ligand_atoms : list of int, str, or None
        The indices of the ligand atoms to restrain, an MDTraj selection string, or a Topography region selection
        string.

    Examples
    --------
    Create the ThermodynamicState.

    >>> from openmmtools import testsystems, states
    >>> system_container = testsystems.LysozymeImplicit()
    >>> system, positions = system_container.system, system_container.positions
    >>> thermodynamic_state = states.ThermodynamicState(system, 300*unit.kelvin)
    >>> sampler_state = states.SamplerState(positions)

    Identify ligand atoms. Topography automatically identify receptor atoms too.

    >>> from yank.yank import Topography
    >>> topography = Topography(system_container.topology, ligand_atoms=range(2603, 2621))

    you can create a completely defined restraint

    >>> restraint = Harmonic(spring_constant=8*unit.kilojoule_per_mole/unit.nanometers**2,
    ...                      restrained_receptor_atoms=[1644, 1650, 1678],
    ...                      restrained_ligand_atoms='resname TMP')

    Or automatically identify the parameters. When trying to impose a restraint
    with undefined parameters, RestraintParameterError is raised.

    >>> restraint = Harmonic()
    >>> try:
    ...     restraint.restrain_state(thermodynamic_state)
    ... except RestraintParameterError:
    ...     print('There are undefined parameters. Choosing restraint parameters automatically.')
    ...     restraint.determine_missing_parameters(thermodynamic_state, sampler_state, topography)
    ...     restraint.restrain_state(thermodynamic_state)
    ...
    There are undefined parameters. Choosing restraint parameters automatically.

    Get standard state correction.

    >>> correction = restraint.get_standard_state_correction(thermodynamic_state)

    """
    def __init__(self, spring_constant=None, **kwargs):
        super(Harmonic, self).__init__(**kwargs)
        self.spring_constant = spring_constant

    @property
    def _are_restraint_parameters_defined(self):
        """bool: True if the restraint parameters are defined."""
        return self.spring_constant is not None

    def _create_restraint_force(self, particles1, particles2):
        """Create a new restraint force between specified atoms.

        Parameters
        ----------
        particles1 : list of int
            Indices of first group of atoms to restraint.
        particles2 : list of int
            Indices of second group of atoms to restraint.

        Returns
        -------
        force : simtk.openmm.Force
           The created restraint force.

        """
        # Create bond force and lambda_restraints parameter to control it.
        if len(particles1) == 1 and len(particles2) == 1:
            # CustomCentroidBondForce works only on 64bit platforms. When the
            # restrained groups only have 1 particle, we can use the standard
            # CustomBondForce so that we can support 32bit platforms too.
            return mmtools.forces.HarmonicRestraintBondForce(spring_constant=self.spring_constant,
                                                             restrained_atom_index1=particles1[0],
                                                             restrained_atom_index2=particles2[0])
        return mmtools.forces.HarmonicRestraintForce(spring_constant=self.spring_constant,
                                                     restrained_atom_indices1=particles1,
                                                     restrained_atom_indices2=particles2)

    def _determine_restraint_parameters(self, thermodynamic_state, sampler_state, topography):
        """Automatically choose a spring constant for the restraint force.

        The spring constant is selected to give 1 kT at one standard deviation
        of receptor atoms about the receptor restrained atom.

        Parameters
        ----------
        thermodynamic_state : openmmtools.states.ThermodynamicState
            The thermodynamic state.
        sampler_state : openmmtools.states.SamplerState
            The sampler state holding the positions of all atoms.
        topography : yank.Topography
            The topography with labeled receptor and ligand atoms.

        """
        # Do not overwrite parameters that are already defined.
        if self.spring_constant is not None:
            return

        receptor_positions = sampler_state.positions[topography.receptor_atoms]
        sigma = pipeline.compute_radius_of_gyration(receptor_positions)

        # Compute corresponding spring constant.
        self.spring_constant = thermodynamic_state.kT / sigma**2

        logger.debug('Spring constant sigma, s = {:.3f} nm'.format(sigma / unit.nanometers))
        logger.debug('K = {:.1f} kcal/mol/A^2'.format(
            self.spring_constant / unit.kilocalories_per_mole * unit.angstroms**2))



# ==============================================================================
# Flat-bottom protein-ligand restraint.
# ==============================================================================

class FlatBottom(RadiallySymmetricRestraint):
    """A receptor-ligand restraint using a flat potential well with harmonic walls.

    An alternative choice to receptor-ligand restraints that uses a flat
    potential inside most of the protein volume with harmonic restraining
    walls outside of this. It can be used to prevent the ligand from
    drifting too far from protein in implicit solvent calculations while
    still exploring the surface of the protein for putative binding sites.

    The restraint is applied between the centroids of two groups of atoms
    that belong to the receptor and the ligand respectively. The centroids
    are determined by a mass-weighted average of the group particles positions.

    More precisely, the energy expression of the restraint is given by

        ``E = lambda_restraints * step(r-r0) * (K/2)*(r-r0)^2``

    where ``K`` is the spring constant, ``r`` is the distance between the
    restrained atoms, ``r0`` is another parameter defining the distance
    at which the restraint is imposed, and ``lambda_restraints``
    is a scale factor that can be used to control the strength of the
    restraint. You can control ``lambda_restraints`` through the class
    :class:`RestraintState`.

    The class supports automatic determination of the parameters left undefined
    in the constructor through :func:`determine_missing_parameters`.

    With OpenCL, groups with more than 1 atom are supported only on 64bit
    platforms.

    Parameters
    ----------
    spring_constant : simtk.unit.Quantity, optional
        The spring constant K (see energy expression above) in units compatible
        with joule/nanometer**2/mole (default is None).
    well_radius : simtk.unit.Quantity, optional
        The distance r0 (see energy expression above) at which the harmonic
        restraint is imposed in units of distance (default is None).
    restrained_receptor_atoms : iterable of int, int, or str, optional
        The indices of the receptor atoms to restrain, an MDTraj DSL expression, or a
        :class:`Topography <yank.Topography>` region name,
        or :func:`Topography Select String <yank.Topography.select>`.
        This can temporarily be left undefined, but ``determine_missing_parameters()``
        must be called before using the Restraint object. The same if a DSL
        expression or Topography region is provided (default is None).
    restrained_ligand_atoms : iterable of int, int, or str, optional
        The indices of the ligand atoms to restrain, an MDTraj DSL expression.
        or a :class:`Topography <yank.Topography>` region name,
        or :func:`Topography Select String <yank.Topography.select>`.
        This can temporarily be left undefined, but ``determine_missing_parameters()``
        must be called before using the Restraint object. The same if a DSL
        expression or Topography region is provided (default is None).

    Attributes
    ----------
    restrained_receptor_atoms : list of int or None
        The indices of the receptor atoms to restrain, an MDTraj selection string, or a Topography region selection
        string.
    restrained_ligand_atoms : list of int or None
        The indices of the ligand atoms to restrain, an MDTraj selection string, or a Topography region selection
        string.

    Examples
    --------
    Create the ThermodynamicState.

    >>> from openmmtools import testsystems, states
    >>> system_container = testsystems.LysozymeImplicit()
    >>> system, positions = system_container.system, system_container.positions
    >>> thermodynamic_state = states.ThermodynamicState(system, 298*unit.kelvin)
    >>> sampler_state = states.SamplerState(positions)

    Identify ligand atoms. Topography automatically identify receptor atoms too.

    >>> from yank.yank import Topography
    >>> topography = Topography(system_container.topology, ligand_atoms=range(2603, 2621))

    You can create a completely defined restraint

    >>> restraint = FlatBottom(spring_constant=0.6*unit.kilocalorie_per_mole/unit.angstroms**2,
    ...                        well_radius=5.2*unit.nanometers, restrained_receptor_atoms=[1644, 1650, 1678],
    ...                        restrained_ligand_atoms='resname TMP')

    or automatically identify the parameters. When trying to impose a restraint
    with undefined parameters, RestraintParameterError is raised.

    >>> restraint = FlatBottom()
    >>> try:
    ...     restraint.restrain_state(thermodynamic_state)
    ... except RestraintParameterError:
    ...     print('There are undefined parameters. Choosing restraint parameters automatically.')
    ...     restraint.determine_missing_parameters(thermodynamic_state, sampler_state, topography)
    ...     restraint.restrain_state(thermodynamic_state)
    ...
    There are undefined parameters. Choosing restraint parameters automatically.

    Get standard state correction.

    >>> correction = restraint.get_standard_state_correction(thermodynamic_state)

    """
    def __init__(self, spring_constant=None, well_radius=None, **kwargs):
        super(FlatBottom, self).__init__(**kwargs)
        self.spring_constant = spring_constant
        self.well_radius = well_radius

    @property
    def _energy_function(self):
        """str: energy expression of the restraint force."""
        return 'step(distance(g1,g2)-r0) * (K/2)*(distance(g1,g2)-r0)^2'

    @property
    def _are_restraint_parameters_defined(self):
        """bool: True if the restraint parameters are defined."""
        return self.spring_constant is not None and self.well_radius is not None

    def _create_restraint_force(self, particles1, particles2):
        """Create a new restraint force between specified atoms.

        Parameters
        ----------
        particles1 : list of int
            Indices of first group of atoms to restraint.
        particles2 : list of int
            Indices of second group of atoms to restraint.

        Returns
        -------
        force : simtk.openmm.Force
           The created restraint force.

        """
        # Create bond force and lambda_restraints parameter to control it.
        if len(particles1) == 1 and len(particles2) == 1:
            # CustomCentroidBondForce works only on 64bit platforms. When the
            # restrained groups only have 1 particle, we can use the standard
            # CustomBondForce so that we can support 32bit platforms too.
            return mmtools.forces.FlatBottomRestraintBondForce(spring_constant=self.spring_constant,
                                                               well_radius=self.well_radius,
                                                               restrained_atom_index1=particles1[0],
                                                               restrained_atom_index2=particles2[0])
        return mmtools.forces.FlatBottomRestraintForce(spring_constant=self.spring_constant,
                                                       well_radius=self.well_radius,
                                                       restrained_atom_indices1=particles1,
                                                       restrained_atom_indices2=particles2)

    def _determine_restraint_parameters(self, thermodynamic_state, sampler_state, topography):
        """Automatically choose a spring constant and well radius.

        The spring constant, is set to 5.92 kcal/mol/A**2, the well
        radius is set at twice the robust estimate of the standard
        deviation (from mean absolute deviation) plus 5 A.

        Parameters
        ----------
        thermodynamic_state : openmmtools.states.ThermodynamicState
            The thermodynamic state.
        sampler_state : openmmtools.states.SamplerState
            The sampler state holding the positions of all atoms.
        topography : yank.Topography
            The topography with labeled receptor and ligand atoms.

        """
        # Determine number of atoms.
        n_atoms = len(topography.receptor_atoms)

        # Check that restrained receptor atoms are in expected range.
        if any(atom_id >= n_atoms for atom_id in self.restrained_receptor_atoms):
            raise ValueError('Receptor atoms {} were selected for restraint, but system '
                             'only has {} atoms.'.format(self.restrained_receptor_atoms, n_atoms))

        # Compute well radius if the user hasn't specified it in the constructor.
        if self.well_radius is None:
            # Get positions of mass-weighted centroid atom.
            # (Working in non-unit-bearing floats for speed.)
            x_unit = sampler_state.positions.unit
            x_restrained_atoms = sampler_state.positions[self.restrained_receptor_atoms, :] / x_unit
            system = thermodynamic_state.system
            masses = np.array([system.getParticleMass(i) / unit.dalton for i in self.restrained_receptor_atoms])
            x_centroid = np.average(x_restrained_atoms, axis=0, weights=masses)

            # Get dimensionless receptor and ligand positions.
            x_receptor = sampler_state.positions[topography.receptor_atoms, :] / x_unit
            x_ligand = sampler_state.positions[topography.ligand_atoms, :] / x_unit

            # Compute maximum square distance from the centroid to any receptor atom.
            # dist2_centroid_receptor[i] is the squared distance from the centroid to receptor atom i.
            dist2_centroid_receptor = pipeline.compute_squared_distances([x_centroid], x_receptor)
            max_dist_receptor = np.sqrt(dist2_centroid_receptor.max()) * x_unit

            # Compute maximum length of the ligand. dist2_ligand_ligand[i][j] is the
            # squared distance between atoms i and j of the ligand.
            dist2_ligand_ligand = pipeline.compute_squared_distances(x_ligand, x_ligand)
            max_length_ligand = np.sqrt(dist2_ligand_ligand.max()) * x_unit

            # Compute the radius of the flat bottom restraint.
            self.well_radius = max_dist_receptor + max_length_ligand/2 + 5*unit.angstrom

        # Set default spring constant if the user hasn't specified it in the constructor.
        if self.spring_constant is None:
            self.spring_constant = 10.0 * thermodynamic_state.kT / unit.angstroms**2

        logger.debug('restraint distance r0 = {:.1f} A'.format(self.well_radius / unit.angstroms))
        logger.debug('K = {:.1f} kcal/mol/A^2'.format(
            self.spring_constant / unit.kilocalories_per_mole * unit.angstroms**2))
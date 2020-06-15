'''
Class containing useful constants and controls which material system is used
for our simulations
'''

class Constants:
    
    def __init__(self, material_system=None):
        '''
        
        Parameters
        ----------
        material_system : string
            String specifying which material system the constant class is for.
            Currently allowed systems are: ["Si/SiO2", "Si/SiGe", "GaAs"]

        Returns
        -------
        None.

        '''
        # Default units are SI
        self.units = "SI"
        
        # Mathematical constants
        self.pi = 3.141592653589793         # pi
        self.h = 6.62607015-34              # Planck's constant [J*s]
        self.hbar = self.h/(2*self.pi)      # Reduced Planck's constant [J*s]
        
        # Physical constants
        self.e = 1.602176634*1E-19          # electron charge [C]
        self.m0 = 9.10938356E-31            # free electron mass [kg]
        self.c = 2.99792458E8               # speed of light [m/s]
        self.muB = 9.274009994E-24          # Bohr magneton [J/T]
        self.eps0 = 8.85418782E-12          # Vacuum permitivity [F/m]
                  
        # Material system constants
        # Supported material systems include [Si/SiO2, Si/SiGe, GaAs]
        self.material_system = material_system
        
        if material_system == "Si/SiO2":
            self.epsR = 7.8                 # Dielectric constant
            self.eps = self.eps0*self.epsR  # Permitivity [F/m]
            self.me = self.m0*0.191         # Effective mass [kg]
        elif material_system == "Si/SiGe":
            self.epsR = 12.375              # Dielectric constant
            self.eps = self.eps0*self.epsR  # Permitivity [F/m]
            self.me = self.m0*0.191         # Effective mass [kg]
        elif material_system == "GaAs":
            self.epsR = 12.4                # Dielectric constant
            self.eps = self.eps0*self.epsR  # Permitivity [F/m]
            self.me = self.m0*0.067         # Effective mass [kg]
        else:
            raise ValueError("Material system either not recognized or defined.\n\
                             Allowed material are: [""Si/SiO2"", ""Si/SiGe"", ""GaAs""]")
                
        def convert_Ry_to_SI(self, data):
            pass
        
        def convert_SI_to_Ry(self, data):
            pass
                
        

from dataclasses import dataclass

@dataclass
class OmicFeatureParser:
    """
    A class to parse and store omic feature information.
    """
    omic_type: str
    feature_name: str

    def __post_init__(self):
        # Ensure the omic_type is in lowercase for consistency
        self.omic_type = self.omic_type.lower()
    def __
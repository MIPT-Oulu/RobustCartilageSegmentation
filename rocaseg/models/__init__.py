from .unet_lext import UNetLext
from .unet_lext_aux import UNetLextAux

from .discr_a import DiscriminatorA


dict_models = {
    'unet_lext': UNetLext,
    'unet_lext_aux': UNetLextAux,

    'discriminator_a': DiscriminatorA,
}

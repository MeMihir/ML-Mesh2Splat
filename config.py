Config = {
    'loss_weights': {
        'position_weight': 1.0,
        'scaling_weight': 1.0,
        'rotation_weight': 1.0,
        'opacity_weight': 1.0,
        'color_weight': 1.0        
    },
    'loss_function': {
        'position_loss': 'mse',
        'scaling_loss': 'mse',
        'rotation_loss': 'mse',
        'opacity_loss': 'mse',
        'color_loss': 'mse'
    }
}
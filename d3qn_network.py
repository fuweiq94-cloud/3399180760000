"""
D3QN Network - Dueling Double Deep Q-Network
Implements the neural network architecture for D3QN agent
"""

import torch
import torch.nn as nn


class D3QN(nn.Module):
    """
    Dueling Double Deep Q-Network
    
    Architecture:
    - Convolutional layers for feature extraction from observation
    - Dueling architecture separating Value stream and Advantage stream
    - Aggregation layer combining V(s) and A(s,a) to get Q(s,a)
    """
    
    def __init__(self, input_dim=9, num_actions=4, fc1_dim=128, fc2_dim=128):
        super(D3QN, self).__init__()
        
        self.input_dim = input_dim
        self.num_actions = num_actions
        
        # Feature extraction layers (can be conv or fully connected depending on observation type)
        # For vision-based observation
        self.conv1 = nn.Conv1d(1, 16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(16)
        self.conv2 = nn.Conv1d(16, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(32)
        
        # Flatten size calculation
        conv_out_size = 32 * 9  # 32 channels, 9 features
        
        # Shared layers
        self.fc_shared = nn.Sequential(
            nn.Linear(conv_out_size, fc1_dim),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        # Dueling streams
        # Value stream: estimates V(s) - scalar value of the state
        self.value_stream = nn.Sequential(
            nn.Linear(fc1_dim, fc2_dim),
            nn.ReLU(),
            nn.Linear(fc2_dim, 1)
        )
        
        # Advantage stream: estimates A(s,a) - advantage of each action
        self.advantage_stream = nn.Sequential(
            nn.Linear(fc1_dim, fc2_dim),
            nn.ReLU(),
            nn.Linear(fc2_dim, num_actions)
        )
    
    def forward(self, x):
        """
        Forward pass through the network
        
        Args:
            x: Input tensor of shape (batch_size, input_dim)
            
        Returns:
            Q-values for all actions of shape (batch_size, num_actions)
        """
        # Reshape for convolutional processing
        batch_size = x.size(0)
        x = x.unsqueeze(1)  # (batch_size, 1, input_dim)
        
        # Convolutional layers
        x = nn.functional.relu(self.bn1(self.conv1(x)))
        x = nn.functional.relu(self.bn2(self.conv2(x)))
        
        # Flatten
        x = x.view(batch_size, -1)  # (batch_size, conv_out_size)
        
        # Shared layers
        x = self.fc_shared(x)
        
        # Separate value and advantage streams
        V = self.value_stream(x)  # (batch_size, 1)
        A = self.advantage_stream(x)  # (batch_size, num_actions)
        
        # Aggregate using the dueling formula:
        # Q(s,a) = V(s) + (A(s,a) - mean(A(s,a')))
        Q = V + (A - A.mean(dim=1, keepdim=True))
        
        return Q
    
    def get_q_values(self, states):
        """Get Q-values for a batch of states"""
        return self.forward(states)
    
    def save(self, path):
        """Save model parameters"""
        torch.save({
            'model_dict': self.state_dict()
        }, path)
    
    def load(self, path):
        """Load model parameters"""
        checkpoint = torch.load(path)
        self.load_state_dict(checkpoint['model_dict'])


def test_d3qn():
    """Test the D3QN network"""
    print("Testing D3QN Network...")
    
    # Create network
    net = D3QN(input_dim=9, num_actions=4)
    print(f"Network created successfully!")
    print(f"Total parameters: {sum(p.numel() for p in net.parameters()):,}")
    
    # Test forward pass
    batch_size = 32
    x = torch.randn(batch_size, 9)
    
    with torch.no_grad():
        output = net(x)
        print(f"\nInput shape: {x.shape}")
        print(f"Output shape: {output.shape}")
        print(f"Sample Q-values:\n{output[:5]}")
    
    # Save/Load test
    net.save('/tmp/d3qn_test.pth')
    new_net = D3QN(input_dim=9, num_actions=4)
    new_net.load('/tmp/d3qn_test.pth')
    print("\n✓ Save/Load test passed!")
    
    print("\n✓ All tests passed!")


if __name__ == '__main__':
    test_d3qn()

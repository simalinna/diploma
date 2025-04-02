from torch import nn

import encoder



class Projector(nn.Module):
    def __init__(self, encoder_dim, projector_dim):
        super().__init__()
        
        self.network = nn.Sequential(
            nn.Linear(encoder_dim, projector_dim),
            nn.BatchNorm1d(projector_dim),
            nn.ReLU(),
            nn.Linear(projector_dim, projector_dim),
            nn.BatchNorm1d(projector_dim),
            nn.ReLU(),
            nn.Linear(projector_dim, projector_dim),
            nn.BatchNorm1d(projector_dim),
            nn.ReLU(),
            nn.Linear(projector_dim, projector_dim)
        )
            
    def forward(self, x):
        return self.network(x)
        

    
class VICReg(nn.Module):
    def __init__(self, encoder_dim, projector_dim, num_classes):
        super().__init__()

        self.encoder = encoder.MyResnet(encoder_dim)
        
        self.projector = Projector(encoder_dim, projector_dim)

        self.linear = nn.Linear(encoder_dim, num_classes)

        # self.linear = nn.Linear(projector_dim, num_classes)
    
    def forward(self, x1, x2, x3):

        emb1 = self.encoder(x1)
        emb2 = self.encoder(x2)
        emb3 = self.encoder(x3)

        rep1 = self.projector(emb1)
        rep2 = self.projector(emb2)
        rep3 = self.projector(emb3)

        pred1 = self.linear(emb1)
        pred2 = self.linear(emb2)
        pred3 = self.linear(emb3)

        # return rep1, rep2, pred1, pred2
        return rep1, rep2, rep3, pred1, pred2, pred3
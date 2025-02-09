import torch
from matplotlib import pyplot as plt

import settings
from torch import nn

from losses import KKTLoss, ProjectedGradientOptimizer
from utils import get_array_of_constraints


class RecNetwork(nn.Module):
    def __init__(self, train_loader, criterion=nn.MSELoss(), num_classes=2, weight_decay=0,
                 real_points_loader=None):
        super(RecNetwork, self).__init__()
        x0 = []
        y0 = []

        for batch_features, batch_labels in train_loader:
            x0.append(batch_features)
            y0.append(batch_labels.squeeze())
        x0 = torch.vstack(x0).to(settings.device).clone().detach()
        y0 = torch.cat(y0).to(settings.device).clone().detach()
        l0 = torch.ones_like(y0, requires_grad=False, device=settings.device, dtype=torch.float32)
        l0 = 1e-3 * l0
        self.y = y0.long()
        self.x = torch.nn.Parameter(torch.tensor(x0), requires_grad=True)
        self.l = torch.nn.Parameter(torch.tensor(l0), requires_grad=True)
        self.criterion = criterion
        self.weight_decay = weight_decay
        self.num_classes = num_classes

        if real_points_loader is not None:
            real_points_x = []
            real_points_y = []
            for batch_features, batch_labels in real_points_loader:
                real_points_x.append(batch_features)
                real_points_y.append(batch_labels)
            self.real_points_x = torch.vstack(real_points_x).to(settings.device).clone().detach()
            self.real_points_y = torch.cat(real_points_y).to(settings.device).clone().detach()
        else:
            self.real_points_x = None
            self.real_points_y = None

    def find_lambdas(self, model):
        self.x = torch.nn.Parameter(self.real_points_x.clone(), requires_grad=True)
        opt_l = torch.optim.Adam([self.l], lr=0.1)
        kkt_loss = KKTLoss(model)
        for epoch in range(settings.find_lambdas_epochs):
            opt_l.zero_grad()
            loss = self.forward(model.eval().to(settings.device), kkt_loss, lambda_loss_reg=1)
            loss.backward()
            opt_l.step()
            if epoch % 1000 == 0:
                print(f"EPOCH {epoch}, LOSS LAMBDA: {loss}")
        print(f"LAMBDAS {self.l}")

    def get_prior_loss(self):
        # return self.unit_ball_loss()
        prior_loss = 0
        prior_loss += 1.2 * (self.x - 1).relu().pow(2).sum()
        prior_loss += 1.2 * (-1 - self.x).relu().pow(2).sum()
        return prior_loss

    def get_lambda_loss(self):
        return (-self.l).relu().pow(2).sum()

    def forward(self, model, kkt_loss, rec_loss_reg=1, prior_loss_reg=0., lambda_loss_reg=0.5):
        model.eval()
        # Define regularization parameters
        # Calculate total loss with regularization parameters
        return (kkt_loss(self.x, self.y, self.l) * rec_loss_reg +
                self.get_prior_loss() * prior_loss_reg +
                self.get_lambda_loss() * lambda_loss_reg)

    def reconstruct_in_polyhedron(self, model, A, b):
        # Extraction phase
        model.eval().to(settings.device)
        kkt_loss = KKTLoss(model)
        opt_x = ProjectedGradientOptimizer([self.x], lr=0.01, A=A, b=b)

        # self.evaluate_extraction(self.x.clone().detach().to(settings.device), self.y.clone().detach(),
        #                          self.real_points_x.clone().detach(), self.real_points_y.clone().detach())

        for epoch in range(1001):
            opt_x.zero_grad()
            loss = self.forward(model.eval().to(settings.device), kkt_loss, lambda_loss_reg=0)
            loss.backward()
            opt_x.step()

        return loss

    def reconstruction(self, model):
        # Extraction phase
        model.eval().to(settings.device)
        kkt_loss = KKTLoss(model)
        opt_x = torch.optim.Adam([self.x], lr=0.001)
        opt_l = torch.optim.SGD([self.l], lr=0.0)

        self.evaluate_extraction(self.x.clone().detach().to(settings.device), self.y.clone().detach(),
                                 self.real_points_x.clone().detach(), self.real_points_y.clone().detach())

        for epoch in range(6001):
            x_copy = self.x.clone().detach().to(settings.device)
            opt_x.zero_grad()
            opt_l.zero_grad()
            loss = self.forward(model.eval().to(settings.device), kkt_loss, lambda_loss_reg=0)
            loss.backward()
            opt_x.step()
            opt_l.step()

            if epoch % 1000 == 0:
                self.evaluate_extraction(x_copy, self.y.clone().detach(),
                                         self.real_points_x.clone().detach(), self.real_points_y.clone().detach())
                for param_group in opt_x.param_groups:
                    hamming_distance_of_constraints = torch.sum(
                        torch.abs(get_array_of_constraints(model, x_copy) - get_array_of_constraints(model, self.x)))
                    hamming_distance_of_constraints_from_training = torch.sum(torch.abs(
                        get_array_of_constraints(model, self.x) - get_array_of_constraints(model,
                                                                                           self.real_points_x.clone().detach())))
                    print(
                        f"loss {loss}, diff {torch.norm(x_copy - self.x)}, "
                        f"Learning Rate: {param_group['lr']}, "
                        f"Distance from previous constraints: {hamming_distance_of_constraints}, "
                        f"Distance from constraints_from_training: {hamming_distance_of_constraints_from_training}")
        print("DONE")

    @staticmethod
    def evaluate_extraction(x, y, real_points_x, real_points_y):
        fig, ax = plt.subplots()
        colors = {0: 'blue', 1: 'red'}  # Color mapping for classes

        # Convert the tensor to a NumPy array if needed
        data = x.cpu().numpy()

        # Iterate through the dataset to get points and their labels
        for i in range(settings.num_samples):
            point, label = data[i], y[i]
            ax.plot(point[0], point[1], 'o', color=colors[label.item()])
            if real_points_x is not None and real_points_y is not None:
                real_point_y = real_points_y[i]
                real_point_x = real_points_x[i]
                ax.plot(real_point_x[0].item(), real_point_x[1].item(), '+', color=colors[real_point_y.item()])

        ax.set_aspect('equal', adjustable='datalim')
        plt.title('Unit Circle Data Points with Alternating Labels')
        plt.xlabel('X coordinate')
        plt.ylabel('Y coordinate')
        plt.grid(True)

        # Display the plot
        plt.show()

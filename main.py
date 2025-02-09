import math

import numpy as np
from torch import nn
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader

import os
import torch
from tqdm import tqdm

import settings
import cvxpy as cp
from datasets import RotatedUnitCircleDataset, RotatedUnitCircleDataset__
from losses import KKTLoss, ProjectedGradientOptimizer
from models.attacked_model import AttackedModel
from models.reconstruction_model import RecNetwork
from utils import get_array_of_constraints, get_minimum_norm_solution, generate_binary_matrices, \
    create_matrix_for_convex_optimization, \
    check_constraints, create_polyhedron_boundaries

if __name__ == '__main__':
    model = AttackedModel(input_dim=settings.input_dim, output_dim=1, hidden_layer_dim=4)
    small_constant_initializer = 1e-3
    with torch.no_grad():
        for layer in model.parameters():
            layer.data = layer.data * small_constant_initializer
    model.to(settings.device)

    dataset = RotatedUnitCircleDataset(num_samples=settings.num_samples, shift_degree=45)
    train_loader = DataLoader(dataset, batch_size=settings.batch_size, shuffle=False)  # dataloader

    optimizer = torch.optim.SGD(model.parameters(), lr=settings.lr)
    scheduler = StepLR(optimizer, step_size=30, gamma=0.7)
    criterion = nn.BCEWithLogitsLoss()

    path_to_saved_model = f"trained_model.pth"
    if os.path.exists(path_to_saved_model):
        print(f"LOAD MODEL")
        model.load_state_dict(torch.load(path_to_saved_model, weights_only=True))
    else:
        for epoch in range(1, settings.epochs + 1):
            model.train()
            for batch_idx, (data, target) in enumerate(train_loader, start=1):
                if data.shape[0] != settings.batch_size:
                    continue

                data, target = data.to(settings.device), target.to(settings.device)
                out = model(data)
                loss = criterion(out, target)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # scheduler.step()
            correct = 0
            model.eval()
            if epoch % 10 == 0:
                with torch.no_grad():
                    for data, labels in train_loader:
                        data, labels = data.to(settings.device), labels.to(settings.device)
                        outputs = model(data)
                        predicted = outputs.data.squeeze()
                        accuracy = ((predicted > 0.0) == labels.squeeze()).float().mean()
                print(
                    f"epoch {epoch}, Loss {loss:>7f} Accuracy {accuracy}, lr {scheduler.optimizer.param_groups[0]['lr']}")
                # if accuracy > 0.99 and epoch % 1000 == 0:
                #     print(f"EPOCH {epoch} PRINT BOUNDARY DECISION")
                #     plot_decision_boundary(model, train_loader)

        torch.save(model.state_dict(), path_to_saved_model)

    # model.plot_decision_boundary(train_loader)
    print("RECONSTRUCTION")

    dataset__ = RotatedUnitCircleDataset__(num_samples=settings.num_samples, shift_degree=45)
    train_loader__ = DataLoader(dataset__, batch_size=settings.batch_size, shuffle=False)
    lambda_rn = RecNetwork(train_loader, real_points_loader=train_loader)
    lambda_rn.find_lambdas(model.eval())
    rn = RecNetwork(train_loader, real_points_loader=train_loader)
    rn.l = lambda_rn.l

    real_points_x = []
    real_points_y = []
    for batch_features, batch_labels in train_loader:
        real_points_x.append(batch_features)
        real_points_y.append(batch_labels)
    real_points_x = torch.vstack(real_points_x).to(settings.device).clone().detach()
    real_points_y = torch.cat(real_points_y).to(settings.device).clone().detach().squeeze()
    real_points_y = 2 * real_points_y - 1

    array_of_constraints = get_array_of_constraints(model, real_points_x)
    print("\n ARRAY OF CONSTRAINTS \n")
    print(array_of_constraints)

    kkt_loss = KKTLoss(model)
    loss_of_training_set = kkt_loss(model.eval()(real_points_x).squeeze(), real_points_y,
                                    lambda_rn.l).detach().cpu().numpy()
    print(f"KKT LOSS OF TRAINING SET IS {loss_of_training_set}")

    # PROJECTED GRADIENT DESCENT
    # print(f"Iterating over {2 ** math.prod(list(array_of_constraints.size()))} matrices")
    # number_of_exception = 0
    # for binary_matrix in tqdm(generate_binary_matrices(*list(array_of_constraints.size()))):
    #     A, b = create_polyhedron_boundaries(model, binary_matrix)
    #     rec_polyhedron_model = RecNetwork(train_loader)
    #     rec_polyhedron_model.l = lambda_rn.l
    #     try:
    #         polyhedron_loss = rec_polyhedron_model.reconstruct_in_polyhedron(model, A, b)
    #     except ValueError:
    #         continue
    #
    #     print("A")

    # QUADRATIC PROGRAMMIING
    print(f"Iterating over {2 ** math.prod(list(array_of_constraints.size()))} matrices")
    number_of_exception = 0
    for binary_matrix in tqdm(generate_binary_matrices(*list(array_of_constraints.size()))):
        binary_matrix = np.array(
        [[0., 0., 0., 0.],
        [0., 0., 1., 1.],
        [1., 1., 1., 1.],
        [0., 1., 1., 0.]]) #TODO: DELETE THIS VARIABLE
        A = create_matrix_for_convex_optimization(model, lambda_rn.l.clone().detach(), binary_matrix,
                                                  real_points_y).detach().cpu().numpy()
        x = cp.Variable(settings.num_samples * settings.input_dim)
        list_of_w = model.state_dict()['layers.0.weight'].detach().cpu().numpy()
        list_of_v = model.state_dict()['layers.1.weight'].detach().cpu().numpy()
        list_of_b = model.state_dict()['layers.0.bias'].detach().cpu().numpy()
        list_of_lambdas = lambda_rn.l.clone().detach().cpu().numpy()

        constraints = []
        for i in range(settings.num_samples):
            point = [x[k] for k in range(i, settings.num_samples * settings.input_dim, settings.num_samples)]
            for j in range(model.state_dict()['layers.0.weight'].size()[0]):
                if binary_matrix[j, i] == 0:
                    constraints.append(cp.vdot(list_of_w[j], point) + list_of_b[j] <= 0)
                else:
                    constraints.append(cp.vdot(list_of_w[j], point) + list_of_b[j] >= 0)

        prob = cp.Problem(cp.Minimize(cp.sum_squares(A @ x - list_of_w.flatten())), constraints)
        try:
            prob_loss = prob.solve(solver='CLARABEL')
        except cp.SolverError:
            number_of_exception += 1
            print(f'{number_of_exception} ERRORS')
            continue
        if prob.status == cp.OPTIMAL:
            print(f"\n BINARY MATRIX: \n {binary_matrix}")
            points = []
            for i in range(settings.num_samples):
                points.append(
                    [x[k].value for k in range(i, settings.num_samples * settings.input_dim, settings.num_samples)])
            points = torch.tensor(points, device=settings.device).to(torch.float)
            points_loss = kkt_loss(model.eval()(points).squeeze(), real_points_y,
                                   torch.tensor(list_of_lambdas, device=settings.device)).detach().cpu().numpy()
            if not math.isclose(prob_loss, points_loss, abs_tol=10 ** -3):
                print(f'PROB LOSS {prob_loss}')
                print(f"POINTS LOSS {points_loss}")

    # MINIMUM NORM SOLUTION

    # print(f"Iterating over {2**math.prod(list(array_of_constraints.size()))} matrices")
    # for binary_matrix in tqdm(generate_binary_matrices(*list(array_of_constraints.size()))):
    #     solutions = get_minimum_norm_solution(create_matrix(model, lambda_rn.l.clone().detach(), binary_matrix, real_points_y),
    #                                           model.state_dict()['layers.0.weight'].clone().detach().view(-1))
    #
    #     constraints = []
    #     for i in range(settings.num_samples):
    #         point = torch.tensor(
    #             [solutions[k] for k in range(i, settings.num_samples * settings.input_dim, settings.num_samples)]).to(
    #             settings.device).clone().detach()
    #         for j in range(model.state_dict()['layers.0.weight'].size()[0]):
    #             constraints.append(check_constraints(point, model.state_dict()['layers.0.weight'][j],
    #                                                  model.state_dict()['layers.0.bias'][j], binary_matrix[j, i]))
    #     if all(constraints):
    #         print(f"\n {binary_matrix} \n")

    # rn.reconstruction(model.eval())

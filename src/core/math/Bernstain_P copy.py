import torch
import numpy as np
import time



class BersteinPoly():
    def __init__(self, n_func:int=8, domain_min=-1, domain_max=1, device='cpu', dtype=torch.float64):
        self.set_number_of_functions(n_func)
        self.domain_min = domain_min
        self.domain_max = domain_max
        self.dtype = dtype
        if device == 'cuda':
             self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = 'cpu'

    def set_points_domain(self, domain_min, domain_max):

        self.domain_min = float(domain_min)
        self.domain_max = float(domain_max)
        # print(f'\033[95mDomain set to [{self.domain_min},{self.domain_max}]\033[0m')

    def set_number_of_functions(self, n_func):
        self.n_func = int(n_func)


    def normalize_points(self, p):
        # print(f'Normalizing points from domain [{self.domain_min},{self.domain_max}]')
        return ((p - self.domain_min)/(self.domain_max-self.domain_min)).reshape(len(p), 3)

    @staticmethod
    def binomial_coefficient(n, k): #C(n,k )
        return torch.exp(torch.lgamma(n + 1) - torch.lgamma(k + 1) - torch.lgamma(n - k + 1))

    def build_bernstein_t(self, t, use_derivative=False): #given t normalized to [0,1] returns the bernstein basis function
        t = torch.clamp(t, min=1e-6, max=1-1e-6)#limit t to be inside min and max
        n = self.n_func - 1
        i = torch.arange(self.n_func, device=t.device)

        comb = self.binomial_coefficient(torch.tensor(n, device=t.device), i)

        phi = comb * (1 - t).unsqueeze(-1) ** (n - i) * t.unsqueeze(-1) ** i #add a dimension to t and i to be able to do the outer product
        #returns Bernstain basis function in order Bi0,---,Bin
        if not use_derivative:
            return phi.float(),None
        else:
            dphi = -comb * (n - i) * (1 - t).unsqueeze(-1) ** (n - i - 1) * t.unsqueeze(-1) ** i + comb * i * (1 - t).unsqueeze(-1) ** (n - i) * t.unsqueeze(-1) ** (i - 1)
            dphi = torch.clamp(dphi, min=-1e4, max=1e4)
            return phi.float(),dphi.float()


    def basis_function_from_3Dpoints(self, p, use_derivative=False):
        p = self.normalize_points(p)
        phi_x, dphi_x = self.build_bernstein_t(p[:, 0], use_derivative)
        phi_y, dphi_y = self.build_bernstein_t(p[:, 1], use_derivative)
        phi_z, dphi_z = self.build_bernstein_t(p[:, 2], use_derivative)
        n = self.n_func
        N = p.shape[0]

        # Base function (contrazione sequenziale)
        phi_xy = torch.einsum("ij,ik->ijk", phi_x, phi_y).reshape(N, n**2)
        phi_xyz = torch.einsum("ij,ik->ijk", phi_xy, phi_z).reshape(N, n**3)

        if not use_derivative:
            return phi_xyz, None

        # Derivate (versione corretta e ottimizzata)
        phi_yz = torch.einsum("ij,ik->ijk", phi_y, phi_z).reshape(N, n**2)

        d_phi_x = torch.einsum("ij,ik->ijk", dphi_x, phi_yz).reshape(N, n**3)
        d_phi_y = torch.einsum("ij,ik->ijk", phi_x, torch.einsum("ij,ik->ijk", dphi_y, phi_z).reshape(N, n**2)).reshape(N, n**3)
        d_phi_z = torch.einsum("ij,ik->ijk", phi_x, torch.einsum("ij,ik->ijk", phi_y, dphi_z).reshape(N, n**2)).reshape(N, n**3)

        d_phi_xyz = torch.stack([d_phi_x, d_phi_y, d_phi_z], dim=-1)
        d_phi_xyz = d_phi_xyz / (self.domain_max - self.domain_min)

        return phi_xyz, d_phi_xyz
        
    






if __name__ == "__main__":
    pass



    # def basis_function_from_3Dpoints_prev(self, p, use_derivative=False) :
    #     #computationally more expensive wrt do the code commented, but more readable
    #     # N = len(p)
    #     # p = self.normalize_points(p).reshape(-1)
    #     # p = ((p - self.domain_min)/(self.domain_max-self.domain_min)).reshape(-1)
    #     # phi,d_phi = self.build_bernstein_t(p,use_derivative)
    #     # phi = phi.reshape(N,3,self.n_func)
    #     # phi_xyz = torch.einsum("ij,ik,il->ijkl", phi[:,0,:], phi[:,1,:], phi[:,2,:]).view(-1,self.n_func**3)
    #     p = self.normalize_points(p)
    #     phi_x, dphi_x = self.build_bernstein_t(p[:,0], use_derivative) # interpolation in x
    #     phi_y, dphi_y = self.build_bernstein_t(p[:,1], use_derivative) # interpolation in y
    #     phi_z, dphi_z = self.build_bernstein_t(p[:,2], use_derivative) # interpolation in z


    #     phi_xyz = torch.einsum("ij,ik,il->ijkl", phi_x, phi_y, phi_z).reshape(-1, self.n_func**3) # creo una griglia di pesi

    #     # print(phi_x[0,0]*phi_y[0,0]*phi_z[0,0])
    #     # print(phi_xyz[0,0,0,0])

    #     # for i in range(self.n_func):
    #     #     p = p.reshape(N,3)
    #     #     print(f'point {p[i]}')
    #     #     print(f'test point {point_test[i]}')
    #     #     print(f'phi_x {phi_x[i]}')
    #     #     print(f'test_phi_x {test_phi_x[i]}')

    #     if use_derivative ==False:
    #         return phi_xyz, None
    #     else:
    #         d_phi_x = torch.einsum("ij,ik->ijk", torch.einsum("ij,ik->ijk",dphi_x, phi_y).reshape(-1,self.n_func**2), phi_z).reshape(-1,self.n_func**3)

    #         d_phi_y = torch.einsum("ij,ik->ijk", torch.einsum("ij,ik->ijk",phi_x,  dphi_y).reshape(-1,self.n_func**2),phi_z).reshape(-1,self.n_func**3)
    #         d_phi_z = torch.einsum("ij,ik->ijk", torch.einsum("ij,ik->ijk",phi_x,  phi_y).reshape(-1,self.n_func**2), dphi_z).reshape(-1,self.n_func**3)
    #         d_phi_xyz = torch.cat((d_phi_x.unsqueeze(-1),d_phi_y.unsqueeze(-1),d_phi_z.unsqueeze(-1)),dim=-1)



    #         d_phi_xyz = d_phi_xyz  /(self.domain_max - self.domain_min)


    #         return phi_xyz, d_phi_xyz


    # def basis_function_from_3Dpoints_prev(self, p, use_derivative=False) :
    #     #computationally more expensive wrt do the code commented, but more readable
    #     # N = len(p)
    #     # p = self.normalize_points(p).reshape(-1)
    #     # p = ((p - self.domain_min)/(self.domain_max-self.domain_min)).reshape(-1)
    #     # phi,d_phi = self.build_bernstein_t(p,use_derivative)
    #     # phi = phi.reshape(N,3,self.n_func)
    #     # phi_xyz = torch.einsum("ij,ik,il->ijkl", phi[:,0,:], phi[:,1,:], phi[:,2,:]).view(-1,self.n_func**3)
    #     p = self.normalize_points(p)
    #     phi_x, dphi_x = self.build_bernstein_t(p[:,0], use_derivative) # interpolation in x
    #     phi_y, dphi_y = self.build_bernstein_t(p[:,1], use_derivative) # interpolation in y
    #     phi_z, dphi_z = self.build_bernstein_t(p[:,2], use_derivative) # interpolation in z


    #     phi_xyz = torch.einsum("ij,ik,il->ijkl", phi_x, phi_y, phi_z).reshape(-1, self.n_func**3) # creo una griglia di pesi

    #     # print(phi_x[0,0]*phi_y[0,0]*phi_z[0,0])
    #     # print(phi_xyz[0,0,0,0])

    #     # for i in range(self.n_func):
    #     #     p = p.reshape(N,3)
    #     #     print(f'point {p[i]}')
    #     #     print(f'test point {point_test[i]}')
    #     #     print(f'phi_x {phi_x[i]}')
    #     #     print(f'test_phi_x {test_phi_x[i]}')

    #     if use_derivative ==False:
    #         return phi_xyz, None
    #     else:
    #         d_phi_x = torch.einsum("ij,ik->ijk", torch.einsum("ij,ik->ijk",dphi_x, phi_y).reshape(-1,self.n_func**2), phi_z).reshape(-1,self.n_func**3)

    #         d_phi_y = torch.einsum("ij,ik->ijk", torch.einsum("ij,ik->ijk",phi_x,  dphi_y).reshape(-1,self.n_func**2),phi_z).reshape(-1,self.n_func**3)
    #         d_phi_z = torch.einsum("ij,ik->ijk", torch.einsum("ij,ik->ijk",phi_x,  phi_y).reshape(-1,self.n_func**2), dphi_z).reshape(-1,self.n_func**3)
    #         d_phi_xyz = torch.cat((d_phi_x.unsqueeze(-1),d_phi_y.unsqueeze(-1),d_phi_z.unsqueeze(-1)),dim=-1)



    #         d_phi_xyz = d_phi_xyz  /(self.domain_max - self.domain_min)


    #         return phi_xyz, d_phi_xyz


    # def predict(self, points:torch.Tensor, get_min:bool=False, get_gradient:bool=False):
    #     if not isinstance(points, torch.Tensor):
    #         raise ValueError("\033[91m" + "The points must be a torch tensor" + "\033[0m")

    #     #print device
    #     points_scaled = (points - self.centroid_offset) / self.scale_factor
    #     # print("\033[38;2;255;165;0m" + f'POINTS SCALED: {points_scaled}' + "\033[0m")
    #     points_scaled, sphere_distances = spherical_projection(points_scaled, r1=(self.domain_max - self.domain_min)/2)

    #     # print(self.n_func)
    #     phi_p, d_phi_p = super().basis_function_from_3Dpoints(points_scaled, get_gradient)
    #     # print("PHI_P", phi_p)
    #     # print("PHI_P", phi_p.shape)
    #     # print("WEIGHTS", self.weights.shape)
    #     sdf = torch.matmul(phi_p, self.weights)
    #     # sdf = torch.einsum('ij,j->i',phi_p,self.weights)

    #     d_sdf = torch.einsum('ijk,j->ik', d_phi_p, self.weights) if get_gradient else torch.zeros((points.shape[0], 3), device=self.device)  # Restituisci solo la SDF

    #     # grad_numeric = torch.zeros((points_scaled.shape[0], 3), device=self.device, dtype=self.dtype)
    #     # epsilon = 1e-3
    #     # if get_gradient:
    #     #     for i in range(3):  # Itera su x, y, z
    #     #             perturbed_points = points_scaled.clone() #punti da perturbare

    #     #             # Perturba solo la i-esima coordinata in avanti
    #     #             perturbed_points[:, i] += epsilon
    #     #             sdf_plus, _ = super().points_to_sdf(perturbed_points)

    #     #             # Perturba solo la i-esima coordinata indietro
    #     #             perturbed_points[:, i] -= 2 * epsilon
    #     #             sdf_minus, _ = super().points_to_sdf(perturbed_points)

    #     #             # Calcola la derivata numerica come differenza centrale
    #     #             grad_numeric[:, i] = ((sdf_plus - sdf_minus) / (2 * epsilon)).squeeze()

    #     # print("Grad numeric", grad_numeric)
    #     # print("Grad analytic", d_sdf)
    #     # exit()


    #     # sdf = sdf + distances_ellipsoid #+ np.linalg.norm(self.centroid_offset)
    #     sdf = sdf  + sphere_distances
    #     sdf = sdf * self.scale_factor

    #     index = None
    #     if get_min:
    #         sdf, index = torch.min(sdf, dim=0)
    #         d_sdf = d_sdf[index]
    #         points = points[index]
    #         # grad_numeric = grad_numeric[index]

    #     return sdf, d_sdf, points

#   def train_with_least_squares(self, point_near, near_sdf, query_points, query_sdf):
#         choice_near = np.random.choice(len(point_near), 18048, replace=False)
#         p_near = torch.from_numpy(point_near[choice_near]).float().to(self.device)
#         sdf_near = torch.from_numpy(near_sdf[choice_near]).float().to(self.device)

#         choice_random = np.random.choice(len(query_points), 11024, replace=False)
#         p_random = torch.from_numpy(query_points[choice_random]).float().to(self.device)
#         sdf_random = torch.from_numpy(query_sdf[choice_random]).float().to(self.device)
#         # point_near = torch.from_numpy(point_near).float().to(self.device)
#         # near_sdf = torch.from_numpy(near_sdf).float().to(self.device)
#         # query_points = torch.from_numpy(query_points).float().to(self.device)
#         # query_sdf = torch.from_numpy(query_sdf).float().to(self.device)
#         # p = torch.cat([point_near, query_points], dim=0)
#         # sdf = torch.cat([near_sdf, query_sdf], dim=0)ù

#         p = torch.cat([p_near, p_random], dim=0)
#         sdf = torch.cat([sdf_near, sdf_random], dim=0)

#         phi_xyz, _ = self.basis_function_from_3Dpoints(p.float(), use_derivative=False)

#         # Risolvi il problema dei minimi quadrati
#         wb, residuals, rank, s = torch.linalg.lstsq(phi_xyz, sdf)

#         return wb

#     def train_with_gradient_descent(self, point_near, near_sdf, query_points, query_sdf, epoches=400, lr=1e-3):
#         # Inizializza i pesi
#         import torch.optim as optim
#         wb = torch.zeros(self.n_func**3, requires_grad=True, device=self.device)
#         optimizer = optim.Adam([wb], lr=lr)

#         for iter in range(epoches):
#             optimizer.zero_grad()

#             # Campiona i punti
#             choice_near = np.random.choice(len(point_near), 1024, replace=False)
#             p_near = torch.from_numpy(point_near[choice_near]).float().to(self.device)
#             sdf_near = torch.from_numpy(near_sdf[choice_near]).float().to(self.device)

#             choice_random = np.random.choice(len(query_points), 64, replace=False)
#             p_random = torch.from_numpy(query_points[choice_random]).float().to(self.device)
#             sdf_random = torch.from_numpy(query_sdf[choice_random]).float().to(self.device)

#             p = torch.cat([p_near, p_random], dim=0)
#             sdf = torch.cat([sdf_near, sdf_random], dim=0)

#             phi_xyz, _ = self.basis_function_from_3Dpoints(p.float(), use_derivative=False)
#             output = torch.matmul(phi_xyz, wb)

#             # Calcola la loss
#             loss = torch.nn.functional.mse_loss(output.squeeze(), sdf)
#             print(f'\033[95m Iteration {iter} loss {loss.item()}\033[0m')

#             # Backpropagation
#             loss.backward()
#             optimizer.step()

#         return wb.detach()


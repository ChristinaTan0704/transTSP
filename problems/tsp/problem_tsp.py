# from cmath import cos
# from torch.utils.data import Dataset
# import torch
# import numpy as np
# import os
# import pickle
# from mapd.lib.transTSP.problems.tsp.state_tsp import StateTSP
# from mapd.lib.transTSP.utils.beam_search import beam_search
# from mapd.utils import coord_to_heatmap


# def clip_float_coord_to_grid(map_size, data):
#     data *= map_size
#     data = torch.round(data)
#     data[data == map_size] = map_size-1
#     data /= map_size
#     return data

# def get_shortest_distace_from_distance_map(gird_num, coord_list, distance_map_list):
    
#     coord_list = (coord_list*gird_num).round().type(torch.int32)
#     coord_list[coord_list==gird_num] = gird_num - 1
#     index_list= torch.range(-1,len(distance_map_list)-2).type(torch.int32).unsqueeze(-1)
#     index_list = torch.cat((index_list, coord_list.cpu()),-1)
#     return distance_map_list[index_list[:,0].tolist(), index_list[:,1].tolist(), index_list[:,2].tolist()].sum()


# class TSP(object):

#     NAME = 'tsp'
#     @staticmethod
#     def get_costs(dataset, pi): 
#         # Check that tours are valid, i.e. contain 0 to n -1
#         print("get L2 loss") # TODO del later
#         assert (
#             torch.arange(pi.size(1), out=pi.data.new()).view(1, -1).expand_as(pi) ==
#             pi.data.sort(1)[0]
#         ).all(), "Invalid tour"
#         # Gather dataset in order of tour
#         d = dataset.gather(1, pi.unsqueeze(-1).expand_as(dataset))
#         # Length is distance (L2-norm of difference) from each next location from its prev and of last from first
#         return (d[:, 1:] - d[:, :-1]).norm(p=2, dim=2).sum(1) + (d[:, 0] - d[:, -1]).norm(p=2, dim=1), None

#     @staticmethod
#     def get_heatmap_costs(opts, dataset, pi): 
#         print("get SD loss") # TODO del later
#         # Check that tours are valid, i.e. contain 0 to n -1
#         input_coord, input_heatmap = dataset
#         if "obs" in opts.problem:
#             _, _, w, h = input_heatmap.shape
#             grid_num = max(w, h)
#         else:
#             grid_num = opts.grid_num

#         assert (
#             torch.arange(pi.size(1), out=pi.data.new()).view(1, -1).expand_as(pi) ==
#             pi.data.sort(1)[0]
#         ).all(), "Invalid tour"
#         # Gather dataset in order of tour
#         d = input_coord.gather(1, pi.unsqueeze(-1).expand_as(input_coord))
#         if "Euclidean" in opts.loss:
#         # Length is distance (L2-norm of difference) from each next location from its prev and of last from first
#             # ! here return the cost for all instances; return size : batch_size * 1; the value represent the tour length for one instance
#             return (d[:, 1:] - d[:, :-1]).norm(p=2, dim=2).sum(1) + (d[:, 0] - d[:, -1]).norm(p=2, dim=1), None
#         elif "Shortest" in opts.loss:
#             cost_list = torch.zeros([len(d)], device=d.device)
#             for index, one_tour in enumerate(pi):
#                 coord_list = input_coord[index][one_tour]
#                 distance_map_list = input_heatmap[index][one_tour]
#                 cost_list[index] = get_shortest_distace_from_distance_map(grid_num, coord_list, distance_map_list)
#             return cost_list, None

#     @staticmethod
#     def make_dataset(*args, **kwargs):
#         return TSPDataset(*args, **kwargs)

#     @staticmethod
#     def make_state(*args, **kwargs):
#         return StateTSP.initialize(*args, **kwargs)

#     @staticmethod
#     def beam_search(input, beam_size, expand_size=None,
#                     compress_mask=False, model=None, max_calc_batch_size=4096):

#         assert model is not None, "Provide model"

#         fixed = model.precompute_fixed(input)

#         def propose_expansions(beam):
#             return model.propose_expansions(
#                 beam, fixed, expand_size, normalize=True, max_calc_batch_size=max_calc_batch_size
#             )

#         state = TSP.make_state(
#             input, visited_dtype=torch.int64 if compress_mask else torch.uint8
#         )

#         return beam_search(state, beam_size, propose_expansions)


# class TSPDataset(Dataset):
    
#     def __init__(self, filename=None, size=50, num_samples=1000000, offset=0, distribution=None, opts=None):
#         super(TSPDataset, self).__init__()
        
#         self.data_set = []
#         self.embed_type = opts.embed
#         self.grid_num = opts.grid_num

#         if filename is not None:
#             assert os.path.splitext(filename)[1] == '.pkl'

#             with open(filename, 'rb') as f:
#                 data = pickle.load(f)
#                 self.data = [torch.FloatTensor(row) for row in (data[offset:offset+num_samples])]

#         else:
#             # Sample points randomly in [0, 1] square
#             self.data = [torch.FloatTensor(size, 2).uniform_(0, 1) for i in range(num_samples)]
        
#         if opts.grid_clip:
#             self.data = [clip_float_coord_to_grid(opts.grid_num, one_sample) for one_sample in self.data]
                    
#         self.coord_heatmap =  coord_to_heatmap( opts.grid_num, opts.heatmap_path)
#         if len(self.data) <= 100:
#             # Try to get more sample 
#             self.data = [self.data[0] for i in range(1024)] 
#         self.size = len(self.data)


#     def __len__(self):
#         return self.size

#     def __getitem__(self, idx):
#         one_instance_heatmap = []
#         for one_data in self.data[idx]:
#             x, y = int(one_data[0]*self.grid_num), int(one_data[1]*self.grid_num)
#             # when using the original generate_data.py data's coordnate can be up to 1
#             if x == self.grid_num:
#                 x -= 1
#             if y == self.grid_num:
#                 y -= 1
#             one_instance_heatmap.append(self.coord_heatmap[x][y])
#         return self.data[idx], torch.FloatTensor(np.array(one_instance_heatmap))


from torch.utils.data import Dataset
import torch
import os
import pickle
from mapd.lib.transTSP.problems.tsp.state_tsp import StateTSP
from mapd.lib.transTSP.utils.beam_search import beam_search

class TSP(object):

    NAME = 'tsp'

    @staticmethod
    def get_costs(dataset, pi):
        # Check that tours are valid, i.e. contain 0 to n -1
        assert (
            torch.arange(pi.size(1), out=pi.data.new()).view(1, -1).expand_as(pi) ==
            pi.data.sort(1)[0]
        ).all(), "Invalid tour"

        # Gather dataset in order of tour
        d = dataset.gather(1, pi.unsqueeze(-1).expand_as(dataset))

        # Length is distance (L2-norm of difference) from each next location from its prev and of last from first
        return (d[:, 1:] - d[:, :-1]).norm(p=2, dim=2).sum(1) + (d[:, 0] - d[:, -1]).norm(p=2, dim=1), None

    @staticmethod
    def make_dataset(*args, **kwargs):
        return TSPDataset(*args, **kwargs)

    @staticmethod
    def make_state(*args, **kwargs):
        return StateTSP.initialize(*args, **kwargs)

    @staticmethod
    def beam_search(input, beam_size, expand_size=None,
                    compress_mask=False, model=None, max_calc_batch_size=4096):

        assert model is not None, "Provide model"

        fixed = model.precompute_fixed(input)

        def propose_expansions(beam):
            return model.propose_expansions(
                beam, fixed, expand_size, normalize=True, max_calc_batch_size=max_calc_batch_size
            )

        state = TSP.make_state(
            input, visited_dtype=torch.int64 if compress_mask else torch.uint8
        )

        return beam_search(state, beam_size, propose_expansions)


class TSPDataset(Dataset):
    
    def __init__(self, filename=None, size=50, num_samples=1000000, offset=0, distribution=None):
        super(TSPDataset, self).__init__()

        self.data_set = []
        if filename is not None:
            assert os.path.splitext(filename)[1] == '.pkl'

            with open(filename, 'rb') as f:
                data = pickle.load(f)
                self.data = [torch.FloatTensor(row) for row in (data[offset:offset+num_samples])]
        else:
            # Sample points randomly in [0, 1] square
            self.data = [torch.FloatTensor(size, 2).uniform_(0, 1) for i in range(num_samples)]

        self.size = len(self.data)

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return self.data[idx]

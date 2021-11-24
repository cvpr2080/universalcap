import torch
import utils


class GridBeamSearch(object):
    def __init__(self, model, max_len: int, eos_idx: int, beam_size: int, num_constraints: int):
        self.model = model
        self.max_len = max_len
        self.eos_idx = eos_idx
        self.beam_size = beam_size
        self.num_constraints = num_constraints
        self.b_s = None
        self.device = None
        self.seq_mask = None
        self.seq_logprob = None
        self.outputs = None
        self.log_probs = None
        self.selected_words = None
        self.all_log_probs = None

    def _reorder_collapsed_tensor(self, x, selected_beam, cur_beam_size, cur_num_constraints, prev_num_constraints):
        shape = [int(sh) for sh in x.shape]

        beam = selected_beam
        for _ in shape[1:]:
            beam = beam.unsqueeze(-1)
        beam_exp = beam.expand(*([self.b_s, cur_num_constraints, self.beam_size] + shape[1:]))
        beam_exp = beam_exp.view([self.b_s, cur_num_constraints * self.beam_size] + shape[1:])

        x_res = x.view(*([self.b_s, cur_beam_size * prev_num_constraints] + shape[1:]))
        x_res = torch.gather(x_res, 1, beam_exp)

        x_res = x_res.view(*([-1, ] + shape[1:]))
        return x_res

    def _reorder_tensor(self, x, selected_beam, cur_beam_size, cur_num_constraints, prev_num_constraints):
        shape = [int(sh) for sh in x.shape]
        x_res = x.view(*([-1, ] + shape[3:]))
        x_res = self._reorder_collapsed_tensor(x_res, selected_beam, cur_beam_size, cur_num_constraints,
                                               prev_num_constraints)
        x_res = x_res.view(*([shape[0], -1, self.beam_size] + shape[3:]))
        return x_res

    def _reorder_state(self, selected_beam, cur_beam_size, cur_num_constraints, prev_num_constraints):
        fn = lambda s: self._reorder_collapsed_tensor(s, selected_beam, cur_beam_size, cur_num_constraints,
                                                      prev_num_constraints)
        return fn

    def _reorder_collapsed(self, x: utils.TensorOrSequence, cur_beam_size: int, cur_num_constraints: int,
                           selected_beam: torch.Tensor, prev_num_constraints: int):
        if isinstance(x, torch.Tensor):
            return self._reorder_collapsed_tensor(x, selected_beam, cur_beam_size, cur_num_constraints,
                                                  prev_num_constraints)
        else:
            new_x = []
            for im in x:
                new_im = self._reorder_collapsed_tensor(im, selected_beam, cur_beam_size, cur_num_constraints,
                                                        prev_num_constraints)
                new_x.append(new_im)
            if isinstance(x, tuple):
                new_x = tuple(new_x)
        return new_x

    def _reorder(self, x: utils.TensorOrSequence, cur_beam_size: int, cur_num_constraints: int,
                 selected_beam: torch.Tensor, prev_num_constraints: int):
        if isinstance(x, torch.Tensor):
            return self._reorder_tensor(x, selected_beam, cur_beam_size, cur_num_constraints, prev_num_constraints)
        else:
            new_x = []
            for im in x:
                new_im = self._reorder_tensor(im, selected_beam, cur_beam_size, cur_num_constraints,
                                              prev_num_constraints)
                new_x.append(new_im)
            if isinstance(x, tuple):
                new_x = tuple(new_x)
        return new_x

    def apply(self, visual: utils.TensorOrSequence, constraints: torch.Tensor, out_size=1, **kwargs):
        self.b_s = utils.get_batch_size(visual)
        self.device = utils.get_device(visual)
        self.seq_mask = torch.ones((self.b_s, 1, 1, 1), device=self.device)
        self.seq_logprob = torch.zeros((self.b_s, 1, 1, 1), device=self.device)
        self.log_probs = []
        self.selected_words = None

        outputs = []
        with self.model.statefulness(self.b_s):
            for t in range(self.max_len):
                visual, outputs = self.iter(t, visual, constraints, outputs, **kwargs)

        # Sort result
        outputs = torch.cat(outputs, -1)
        log_probs = torch.cat(self.log_probs, -1)
        outs = []
        outs_log_probs = []
        for c in range(self.num_constraints+1):
            seq_logprob, sort_idxs = torch.sort(self.seq_logprob[:, c], 1, descending=True)
            this_out = torch.gather(outputs[:, c], 1, sort_idxs.expand(self.b_s, self.beam_size, self.max_len))
            this_out = this_out.contiguous()[:, :out_size].unsqueeze(1)
            outs.append(this_out)
            this_log_probs = torch.gather(log_probs[:, c], 1, sort_idxs.expand(self.b_s, self.beam_size, self.max_len))
            this_log_probs = this_log_probs.contiguous()[:, :out_size].unsqueeze(1)
            outs_log_probs.append(this_log_probs)

        outs = torch.cat(outs, 1).contiguous()
        outs_log_probs = torch.cat(outs_log_probs, 1).contiguous()

        if out_size == 1:
            outs = outs.squeeze(2)
            outs_log_probs = outs_log_probs.squeeze(2)

        return outs, outs_log_probs

    def select(self, t, candidate_logprob, constraints, outputs, cur_num_constraints):
        # candidate_logprob (b_s, prev_num_constraints, cur_beam_size, V)
        # constraints (b_s, num_constraints, 1)
        b_s, prev_num_constraints, cur_beam_size, V = candidate_logprob.shape
        device = candidate_logprob.device
        offset = -999 * torch.ones((b_s, 1, cur_beam_size, V), device=device)

        # Aggiungo alla maschera tutti i vincoli
        constr_select_mask = torch.zeros((b_s, self.num_constraints, V), device=device)
        constr_select_mask.scatter_(-1, constraints.long(), 1)  # (b_s, cur_num_constraints, V)
        constr_select_mask = torch.sum(constr_select_mask, 1, keepdim=True).unsqueeze(2)  # (b_s, 1, 1, V): 1 i vincoli, 0 i non vincoli
        constr_select_mask = torch.clamp(constr_select_mask, 0, 1)
        constr_ban_mask = 1-constr_select_mask

        # Tolgo dalla maschera i vincoli già verificati
        if len(outputs) > 0:
            outputs = torch.cat(outputs, -1)
            constr_select_mask = constr_select_mask.expand((-1, prev_num_constraints, cur_beam_size, V))
            constr_select_mask = constr_select_mask.scatter(-1, outputs.long(), 0)

        candidate_logprob_below = candidate_logprob
        candidate_logprob_below = candidate_logprob_below * constr_select_mask - 999 * (1 - constr_select_mask)
        candidate_logprob = candidate_logprob * constr_ban_mask - 999 * (1-constr_ban_mask)

        if cur_num_constraints > prev_num_constraints:
            candidate_logprob = torch.cat((candidate_logprob, offset), 1)
            candidate_logprob_below = torch.cat((offset, candidate_logprob_below), 1)
        else:
            candidate_logprob_below = torch.cat((offset, candidate_logprob_below[:, :-1]), 1)

        candidate_logprob = torch.cat([candidate_logprob_below, candidate_logprob], 2)  # (b_s,
        # cur_num_constraints, 2*beam_size, V)

        selected_logprob, selected_idx = torch.sort(candidate_logprob.view(b_s, cur_num_constraints, -1), -1,
                                                    descending=True)
        # (b_s, cur_num_constraints, 2*beam_size*V), (b_s, cur_num_constraints, 2*beam_size*V)

        selected_logprob, selected_idx = selected_logprob[:, :, :self.beam_size], selected_idx[:, :, :self.beam_size]
        # (b_s, cur_num_constraints, beam_size), (b_s, cur_num_constraints, beam_size)

        return selected_idx, selected_logprob

    def iter(self, t: int, visual: utils.TensorOrSequence, constraints: torch.Tensor, outputs, **kwargs):
        cur_beam_size = 1 if t == 0 else self.beam_size
        prev_num_constraints = max(min(t, self.num_constraints + 1), 1)
        cur_num_constraints = min(t + 1, self.num_constraints + 1)

        word_logits = self.model.step(t, self.selected_words, visual, **kwargs)
        word_logits = word_logits.view(self.b_s, prev_num_constraints, cur_beam_size, -1)
        word_logprob = torch.log_softmax(word_logits, dim=-1)
        candidate_logprob = self.seq_logprob + word_logprob  # (b_s, numC, beam_size, V)

        # Mask sequence if it reaches EOS
        if t > 0:
            mask = (self.selected_words.view(self.b_s, prev_num_constraints, cur_beam_size) != self.eos_idx).type(visual.dtype).unsqueeze(-1)
            self.seq_mask = self.seq_mask * mask
            word_logprob = word_logprob * self.seq_mask.expand_as(word_logprob)
            old_seq_logprob = self.seq_logprob.expand_as(candidate_logprob).contiguous()
            old_seq_logprob[:, :, :, 1:] = -999
            candidate_logprob = self.seq_mask * candidate_logprob + old_seq_logprob * (1 - self.seq_mask)

        selected_idx, selected_logprob = self.select(t, candidate_logprob, constraints, outputs, cur_num_constraints)
        # (b_s, numC, beam_size), (b_s, numC, beam_size)
        selected_beam = torch.floor_divide(selected_idx, candidate_logprob.shape[-1])  # va da 0 a 2*num_beams-1
        selected_words = selected_idx - selected_beam * candidate_logprob.shape[-1]
        selected_beam = selected_beam - cur_beam_size  # va da -num_beams a num_beams-1
        selected_beam = selected_beam + (torch.arange(cur_num_constraints).to(visual.device)).unsqueeze(0).unsqueeze(
            -1) * cur_beam_size
        selected_beam = torch.clamp(selected_beam, min=0)

        self.model.apply_to_states(
            self._reorder_state(selected_beam, cur_beam_size, cur_num_constraints, prev_num_constraints))
        visual = self._reorder_collapsed(visual, cur_beam_size, cur_num_constraints, selected_beam, prev_num_constraints)

        self.seq_logprob = selected_logprob.unsqueeze(-1)
        self.seq_mask = self._reorder(self.seq_mask, cur_beam_size, cur_num_constraints, selected_beam, prev_num_constraints)
        outputs = self._reorder(outputs, cur_beam_size, cur_num_constraints, selected_beam, prev_num_constraints)
        outputs.append(selected_words.unsqueeze(-1))

        this_word_logprob = self._reorder(word_logprob, cur_beam_size, cur_num_constraints, selected_beam, prev_num_constraints)
        this_word_logprob = torch.gather(this_word_logprob, 3, selected_words.unsqueeze(-1))
        self.log_probs = self._reorder(self.log_probs, cur_beam_size, cur_num_constraints, selected_beam, prev_num_constraints)
        self.log_probs.append(this_word_logprob)
        self.selected_words = selected_words.view(-1, 1)

        return visual, outputs

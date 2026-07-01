from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass

from . import config


@dataclass
class NodeGene:
    id: int
    kind: str  # 'in', 'hidden', 'out'


@dataclass
class ConnGene:
    in_node: int
    out_node: int
    weight: float
    enabled: bool
    innovation: int


class FeedForwardNetwork:
    def __init__(self, nodes: dict[int, NodeGene], conns: list[ConnGene], input_ids: list[int], output_ids: list[int]):
        self.nodes = nodes
        self.conns = [c for c in conns if c.enabled]
        self.input_ids = input_ids
        self.output_ids = output_ids

        # Build adjacency for evaluation
        self.incoming: dict[int, list[ConnGene]] = {nid: [] for nid in nodes}
        for c in self.conns:
            self.incoming[c.out_node].append(c)

        # Precompute evaluation order once.
        # forward() is called thousands of times; sorting every call is very slow.
        self.node_ids_sorted = sorted(self.nodes.keys())


    @staticmethod
    def _act(x: float) -> float:
        # Sigmoid-ish but stable
        return 1.0 / (1.0 + math.exp(-x))

    def forward(self, inputs: list[float]) -> list[float]:
        # Compute activations by repeatedly topologically sorting.
        # For this quick prototype, we assume feedforward by construction:
        # mutations only add connections from lower to higher node id.
        values: dict[int, float] = {}
        for nid, v in zip(self.input_ids, inputs):
            values[nid] = float(v)

        # Evaluate hidden + output nodes in increasing id order
        for nid in self.node_ids_sorted:
            if nid in values:
                continue

            incoming = self.incoming.get(nid, [])
            s = 0.0
            for c in incoming:
                s += c.weight * values.get(c.in_node, 0.0)
            values[nid] = self._act(s)

        return [values.get(nid, 0.0) for nid in self.output_ids]


class Genome:
    def __init__(
        self,
        rng: random.Random,
        input_size: int,
        output_size: int,
        nodes: dict[int, NodeGene] | None = None,
        conns: list[ConnGene] | None = None,
        next_node_id: int | None = None,
        next_innovation: int | None = None,
    ) -> None:
        self.rng = rng
        self.input_size = input_size
        self.output_size = output_size

        if nodes is None or conns is None:
            self.nodes: dict[int, NodeGene] = {}
            self.conns: list[ConnGene] = []

            self.input_ids = list(range(0, input_size))
            self.output_ids = list(range(input_size, input_size + output_size))

            for nid in self.input_ids:
                self.nodes[nid] = NodeGene(nid, 'in')
            for nid in self.output_ids:
                self.nodes[nid] = NodeGene(nid, 'out')

            self.next_node_id = input_size + output_size
            self.next_innovation = 0

            # Fully connect inputs -> outputs initially
            for i in self.input_ids:
                for o in self.output_ids:
                    self.conns.append(
                        ConnGene(
                            in_node=i,
                            out_node=o,
                            weight=self.rng.uniform(-1.0, 1.0),
                            enabled=True,
                            innovation=self.next_innovation,
                        )
                    )
                    self.next_innovation += 1

        else:
            self.nodes = nodes
            self.conns = conns
            self.input_ids = list(sorted([nid for nid, n in nodes.items() if n.kind == 'in']))
            self.output_ids = list(sorted([nid for nid, n in nodes.items() if n.kind == 'out']))
            self.next_node_id = next_node_id if next_node_id is not None else (max(nodes.keys()) + 1)
            self.next_innovation = next_innovation if next_innovation is not None else (max([c.innovation for c in conns], default=-1) + 1)

    def copy(self) -> 'Genome':
        nodes = {nid: NodeGene(n.id, n.kind) for nid, n in self.nodes.items()}
        conns = [ConnGene(c.in_node, c.out_node, c.weight, c.enabled, c.innovation) for c in self.conns]
        return Genome(
            rng=random.Random(self.rng.random()),
            input_size=self.input_size,
            output_size=self.output_size,
            nodes=nodes,
            conns=conns,
            next_node_id=self.next_node_id,
            next_innovation=self.next_innovation,
        )

    def build_network(self) -> FeedForwardNetwork:
        return FeedForwardNetwork(self.nodes, self.conns, self.input_ids, self.output_ids)

    def mutate(self) -> None:
        # Weight mutation
        for c in self.conns:
            if self.rng.random() < config.WEIGHT_MUT_PROB:
                if self.rng.random() < config.WEIGHT_REPLACE_PROB:
                    c.weight = self.rng.uniform(-config.WEIGHT_REPLACE_RANGE, config.WEIGHT_REPLACE_RANGE)
                else:
                    c.weight += self.rng.gauss(0.0, config.WEIGHT_PERTURB_STD)

        # Add connection
        if self.rng.random() < config.ADD_CONN_PROB:
            self._add_connection()

        # Add node
        if self.rng.random() < config.ADD_NODE_PROB:
            self._add_node()

    def _add_connection(self) -> None:
        # Try random pairs ensuring feedforward direction (lower id -> higher id)
        tries = 30
        node_ids = list(self.nodes.keys())
        for _ in range(tries):
            a = self.rng.choice(node_ids)
            b = self.rng.choice(node_ids)
            if a == b:
                continue
            if a > b:
                continue  # enforce direction
            if self.nodes[a].kind == 'out':
                continue
            if self.nodes[b].kind == 'in':
                continue

            # Check if connection exists
            for c in self.conns:
                if c.in_node == a and c.out_node == b and c.enabled:
                    return

            self.conns.append(
                ConnGene(
                    in_node=a,
                    out_node=b,
                    weight=self.rng.uniform(-1.0, 1.0),
                    enabled=True,
                    innovation=self.next_innovation,
                )
            )
            self.next_innovation += 1
            return

    def _add_node(self) -> None:
        # Split an enabled connection
        enabled = [c for c in self.conns if c.enabled]
        if not enabled:
            return
        conn = self.rng.choice(enabled)
        conn.enabled = False

        new_node_id = self.next_node_id
        self.next_node_id += 1
        self.nodes[new_node_id] = NodeGene(new_node_id, 'hidden')

        # Add connections: in -> new, new -> out
        c1 = ConnGene(
            in_node=conn.in_node,
            out_node=new_node_id,
            weight=1.0,
            enabled=True,
            innovation=self.next_innovation,
        )
        self.next_innovation += 1

        c2 = ConnGene(
            in_node=new_node_id,
            out_node=conn.out_node,
            weight=conn.weight,
            enabled=True,
            innovation=self.next_innovation,
        )
        self.next_innovation += 1

        self.conns.append(c1)
        self.conns.append(c2)

    def crossover(self, other: 'Genome') -> 'Genome':
        # Very simple crossover: assume self is fitter; inherit all matching/disjoint from fitter.
        # Innovation numbers act like alignment keys.
        by_innov_self = {c.innovation: c for c in self.conns}
        by_innov_other = {c.innovation: c for c in other.conns}
        all_innov = sorted(set(by_innov_self.keys()) | set(by_innov_other.keys()))

        child_conns: list[ConnGene] = []
        for innov in all_innov:
            c1 = by_innov_self.get(innov)
            c2 = by_innov_other.get(innov)
            if c1 and c2:
                chosen = c1 if self.rng.random() < 0.5 else c2
                enabled = c1.enabled and c2.enabled
                # if either is disabled, enable with 25% chance
                if (not c1.enabled) or (not c2.enabled):
                    enabled = self.rng.random() < 0.25
                child_conns.append(
                    ConnGene(
                        in_node=chosen.in_node,
                        out_node=chosen.out_node,
                        weight=chosen.weight,
                        enabled=enabled,
                        innovation=innov,
                    )
                )
            elif c1:
                child_conns.append(
                    ConnGene(c1.in_node, c1.out_node, c1.weight, c1.enabled, c1.innovation)
                )
            elif c2:
                child_conns.append(
                    ConnGene(c2.in_node, c2.out_node, c2.weight, c2.enabled, c2.innovation)
                )

        # Nodes: union of node genes referenced by connections + keep input/output kinds.
        nodes = {}
        for nid in self.nodes:
            nodes[nid] = NodeGene(self.nodes[nid].id, self.nodes[nid].kind)
        for nid in other.nodes:
            if nid not in nodes:
                nodes[nid] = NodeGene(other.nodes[nid].id, other.nodes[nid].kind)

        child = Genome(
            rng=random.Random(self.rng.random()),
            input_size=self.input_size,
            output_size=self.output_size,
            nodes=nodes,
            conns=child_conns,
            next_node_id=max(nodes.keys()) + 1 if nodes else 0,
            next_innovation=max([c.innovation for c in child_conns], default=-1) + 1,
        )
        return child

    def to_dict(self) -> dict:
        return {
            'input_size': self.input_size,
            'output_size': self.output_size,
            'nodes': [{'id': n.id, 'kind': n.kind} for n in self.nodes.values()],
            'conns': [
                {
                    'in_node': c.in_node,
                    'out_node': c.out_node,
                    'weight': c.weight,
                    'enabled': c.enabled,
                    'innovation': c.innovation,
                }
                for c in self.conns
            ],
            'next_node_id': self.next_node_id,
            'next_innovation': self.next_innovation,
        }

    @staticmethod
    def from_dict(d: dict, rng: random.Random | None = None) -> 'Genome':
        rng = rng or random.Random(0)
        nodes = {n['id']: NodeGene(n['id'], n['kind']) for n in d['nodes']}
        conns = [
            ConnGene(
                c['in_node'],
                c['out_node'],
                c['weight'],
                c['enabled'],
                c['innovation'],
            )
            for c in d['conns']
        ]
        return Genome(
            rng=rng,
            input_size=d['input_size'],
            output_size=d['output_size'],
            nodes=nodes,
            conns=conns,
            next_node_id=d['next_node_id'],
            next_innovation=d['next_innovation'],
        )


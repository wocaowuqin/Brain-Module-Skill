"""
envs/modules/high_level_controller.py
====================================
楂樺眰浜や簰鎺у埗鍣?- TA-HRL v4 (HQDQN 瀵归綈鐗?
====================================

銆愭ā鍧楀畾浣嶃€?
鏈ā鍧楁槸 TA-HRL 绯荤粺鐨勯珮灞傜幆澧冩帴鍙ｅ眰锛岃礋璐ｆ妸 HRL_Coordinator 鍙戝嚭鐨勯珮灞傚姩浣滐紙鐩爣鑺傜偣 ID锛?
杞寲涓虹幆澧冪姸鎬佸彉鏇达紝骞跺悜鍗忚皟鍣ㄨ繑鍥炲姩浣滄槸鍚﹀悎娉曘€佸綋鍓嶄换鍔℃槸鍚﹀畬鎴愮瓑淇℃伅銆?
瀹冧笉瀛樺偍绛栫暐鍙傛暟锛屼篃涓嶅仛璁粌锛屽彧鍋氱姸鎬佺鐞嗗拰鎺ュ彛閫傞厤銆?

銆愪袱涓叧閿嚱鏁扮殑璇箟銆?
  set_high_level_goal(high_action_idx, target_node_id, start_node_id)
    - 鎶?target_node_id 鍐欏叆 env.current_subgoal_node / current_deployment_target /
      current_target_node锛屽喅瀹氫綆灞傝璧板悜鍝噷
    - 鍚屾 start_node锛歞est 闃舵寮哄埗浠?last_vnf 鍑哄彂锛屼繚璇?Source鈫扸NF鈫扗est 涓诲共涓嶆柇瑁?
    - 瀵?VNF 闃舵鍋氭渶鍚庝竴閬?DC 鑺傜偣鎷︽埅锛岄槻姝㈡妸鏅€氳妭鐐瑰綋閮ㄧ讲鐩爣

  step_high_level(action_idx)
    - 鎶?action_idx 鐩存帴褰撶洰鏍囪妭鐐?ID锛坕nt(action_idx) = target_node锛?
    - 妫€鏌ョ洰鏍囨槸鍚﹀凡杩為€氾細宸茶繛閫氬垯鍒ゆ柇鏄惁鍏ㄩ儴瀹屾垚锛堣Е鍙?episode 缁撴潫锛夋垨鎴柇褰撳墠瀛愮洰鏍?
    - 鐩镐綅鍚屾锛氬鏋?VNF 宸插叏閮ㄥ畬鎴愪絾 phase 杩樻槸 vnf_deployment锛屽己鍒跺垏鎹㈠埌
      destination_connection锛岄槻姝?mask 鍦ㄦ棫 phase 涓嬭鍒?

銆恆ction mask 鐢熸垚锛坓et_high_level_action_mask锛夈€?
  鏍规嵁褰撳墠 phase 鍒嗕袱濂楅€昏緫锛?
    VNF 闃舵锛氬彧寮€鏀炬湁瓒冲 CPU/MEM 鐨?DC 鑺傜偣
      - 棰勮绠?prev_vnf 鍒板悇 DC 鐨勮烦鏁帮紝灞忚斀璺濈杩囪繙鐨勫€欓€夛紙鍑忓皯鏃犳晥閫夋嫨锛?
      - 灞忚斀宸查儴缃茶繃褰撳墠 VNF 鐨勮妭鐐癸紙闃叉閲嶅閮ㄧ讲锛?
      - BW 瀛ゅ矝杩囨护锛氬鏋滄煇 DC 鐨勬墍鏈夊叆鍚戦摼璺?BW 涓嶈冻锛岀洿鎺ュ睆钄?
    Dest 闃舵锛氬彧寮€鏀惧皻鏈繛閫氱殑鐩殑鍦拌妭鐐?
      - BW 瀛ゅ矝杩囨护锛氬睆钄芥墍鏈夊叆鍚戦摼璺?BW 涓嶈冻鐨勭洰鐨勫湴

銆愮姸鎬佸浘鏋勫缓锛坓et_high_level_state_graph锛夈€?
  杈撳嚭 PyG Data 瀵硅薄锛屽寘鍚細
    - 鑺傜偣鐗瑰緛锛?2 缁达級锛欳PU/MEM 浣欓噺銆乂NF 閮ㄧ讲鐘舵€侊紙hvt锛夈€佺洰鐨勫湴杩為€氱姸鎬併€?
      鑺傜偣绫诲瀷銆佸綋鍓嶄綅缃爣璁般€乸hase 鏍囧織绛?
    - 杈圭壒寰侊細BW 浣欓噺銆佹槸鍚︽爲杈广€侀摼璺潈閲嶇瓑
    - 鍏ㄥ眬灞炴€э細璇锋眰 BW/VNF 闇€姹傘€佽繘搴︺€乸hase 绛?
  渚?HRLAgent 鐨?GNN encoder 缂栫爜鎴愬浘宓屽叆锛屼綔涓洪珮灞?policy 鐨勮緭鍏?

銆愪富瑕佸嚱鏁扮储寮曘€?
  set_high_level_goal()           璁剧疆瀛愮洰鏍囷紝鍚屾 env 鐘舵€佸瓧娈?
  step_high_level()               鎵ц楂樺眰 step锛屽垽鏂瓙鐩爣瀹屾垚/鎴柇/缁х画
  get_high_level_action_mask()    鐢熸垚鍚堟硶鐩爣鍊欓€?mask锛圴NF闃舵/Dest闃舵涓ゅ閫昏緫锛?
  get_high_level_state_graph()    鏋勫缓 PyG 鍥剧姸鎬侊紝浣滀负楂樺眰 policy 杈撳叆
  _is_all_tasks_completed()       鍒ゆ柇 VNF + Dest 鏄惁鍏ㄩ儴瀹屾垚锛岃繑鍥?(bool, status_str)
  _is_valid_node()                鑺傜偣鍚堟硶鎬ф鏌ワ紙ID 鑼冨洿 + 鍙揪鎬э級
  _get_hop_distance()             甯︾紦瀛樼殑閫?episode 璺虫暟鏌ヨ锛岄伩鍏?mask 寰幆涓噸寤哄浘

銆愪笌鍏朵粬妯″潡鐨勪緷璧栧叧绯汇€?
  鈫?HRL_Coordinator    璋冪敤 set_high_level_goal() / step_high_level() 椹卞姩楂樺眰鎵ц
  鈫?LowLevelController 渚濊禆 env.current_deployment_target / current_target_node 纭畾瀛愮洰鏍?
  鈫?SFCEnv             璇诲啓 current_phase / next_vnf_idx / current_tree / chain_nodes 绛夌姸鎬?
  鈫?AllResourceManager 閫氳繃 env.resource_mgr 妫€鏌?DC 鑺傜偣 CPU/MEM 鍙敤閲忓拰閾捐矾 BW

銆愬叧閿璁″喅绛栬褰曘€?
  - step_high_level 鐨?action_idx 鐩存帴浣滀负鑺傜偣 ID锛堜笉鏄?mask 鍘嬬缉鍚庣殑绱㈠紩锛夛紝
    涓?HRL_Coordinator 鐨?actual_high_action = int(target_node) 淇濇寔涓€鑷?
  - phase 鍚屾鍦?step_high_level 鏈€鍓嶉潰瀹屾垚锛屼繚璇?mask 鍦ㄦ纭?context 涓嬭绠?
  - _get_hop_distance 鎸?episode锛坈urrent_request id锛夌紦瀛橈紝閬垮厤 mask 寰幆
    閲屾瘡娆¤皟鐢ㄩ兘閲嶅缓 networkx 鍥撅紙涔嬪墠鏄?Bug7 鐨勬牴鍥狅級
  - 涓変釜宸插垹闄ょ殑鍐椾綑鍑芥暟锛歘is_all_completed()锛堣 _is_all_tasks_completed 鏇夸唬锛夈€?
    _validate_start_node()锛堣捣鐐规帹瀵煎凡鍦?Coordinator 淇濊瘉锛夈€?
    _get_total_vnf_progress()锛堣皟鐢ㄦ柟鐩存帴璇?next_vnf_idx 鍗冲彲锛?
"""

import numpy as np
import torch
import logging
import networkx as nx
import heapq
from torch_geometric.data import Data

logger = logging.getLogger(__name__)
try:
    from .controller_shared_helper import ControllerSharedHelper
except Exception:
    try:
        from controller_shared_helper import ControllerSharedHelper
    except Exception:
        from .controller_shared_helper import ControllerSharedHelper


class HighLevelController:
    """
     楂樺眰浜や簰鎺у埗鍣?- HQDQN 瀵归綈浼樺寲鐗?
    """

    def __init__(self, env):
        self.env = env
        self.shared = ControllerSharedHelper(env)
        # Static topology is independent of the request/resource snapshot.
        # Cache its edge order and tensor; state construction only refreshes
        # dynamic bandwidth/tree attributes.
        self._static_topology_ready = False
        self._static_edge_uv = []
        self._static_edge_index = None
        self._static_edge_hop_weight = []
        self._ensure_static_topology_cache()
        logger.debug(" HighLevelController initialized (HQDQN瀵归綈鐗?")

    def _find_ordered_stage_path(self, start, target, bw_req, blocked_nodes=None):
        """Find a directed, BW-feasible extension of the current SFC spine.

        During VNF placement, a new stage must extend from the previous stage.
        It must not pass through a destination or re-enter an existing tree node,
        because either case can place traffic before the last VNF or create a
        second parent in the canonical directed tree.
        """
        try:
            start, target = int(start), int(target)
        except (TypeError, ValueError):
            return None
        if start == target:
            return [start]

        blocked = {int(node) for node in (blocked_nodes or ())}
        tree_edges = {
            (int(u), int(v))
            for u, v in self.shared.get_positive_tree_edge_set()
        }
        existing_tree_nodes = {
            node for edge in tree_edges for node in edge
        }
        if target in existing_tree_nodes and target != start:
            return None
        for u, v in tree_edges:
            blocked.add(u)
            blocked.add(v)
        blocked.discard(start)
        blocked.discard(target)

        queue = [start]
        seen = {start}
        parent = {start: None}
        while queue:
            node = queue.pop(0)
            try:
                neighbors = self.env.resource_mgr.get_neighbors(node)
            except Exception:
                return None
            for neighbor in neighbors:
                try:
                    neighbor = int(neighbor)
                except (TypeError, ValueError):
                    continue
                if neighbor in seen or neighbor in blocked:
                    continue
                edge = (node, neighbor)
                try:
                    available = float(
                        self.env.resource_mgr.pool.get_available_bandwidth(*edge)
                    )
                except Exception:
                    continue
                if edge not in tree_edges and available + 1e-9 < float(bw_req):
                    continue
                parent[neighbor] = node
                if neighbor == target:
                    path = [target]
                    while parent[path[-1]] is not None:
                        path.append(parent[path[-1]])
                    path.reverse()
                    return path
                seen.add(neighbor)
                queue.append(neighbor)
        return None

    def _has_ordered_stage_path(self, start, target, bw_req, blocked_nodes=None):
        return self._find_ordered_stage_path(
            start, target, bw_req, blocked_nodes=blocked_nodes
        ) is not None

    def _can_complete_destination_tree(
            self, root, destinations, bw_req, blocked_spine, beam_width=64):
        """Check joint receiver feasibility after a hypothetical VNF spine.

        Independent root-to-receiver reachability is insufficient: attaching an
        early receiver can consume a cut node needed by the remaining branches.
        This bounded decoder therefore evaluates receiver orders against one
        evolving directed tree, matching the online destination executor.
        """
        beam_width = max(1, int(getattr(
            self.env, '_online_destination_beam_width', beam_width
        ) or beam_width))
        root = int(root)
        pending = tuple(sorted(int(node) for node in destinations))
        if not pending:
            return True

        occupied = {int(node) for node in blocked_spine}
        occupied.add(root)
        pool = self.env.resource_mgr.pool

        adjacency = {node: [] for node in range(self.env.n)}
        for u in range(self.env.n):
            try:
                neighbors = self.env.resource_mgr.get_neighbors(u)
            except Exception:
                neighbors = ()
            for v in neighbors:
                u, v = int(u), int(v)
                available = float(pool.get_available_bandwidth(u, v))
                if available + 1e-9 < float(bw_req):
                    continue
                capacity = float(pool.bw_cap.get((u, v), 1.0))
                utilization = max(
                    0.0, min(1.0, 1.0 - available / max(1.0, capacity))
                )
                adjacency[u].append((v, 1.0 + utilization))

        def find_attachment(target, remaining, tree_nodes, full_nodes, connected):
            # A receiver switch remains a valid multicast replication point
            # after local delivery.  Once the complete VNF chain has reached
            # it, one output may terminate at the receiver while another
            # continues towards a later receiver.
            anchors = set(full_nodes)
            if not anchors:
                return None
            if int(target) in anchors:
                return [int(target)]
            blocked_terminals = set(remaining) - {int(target)}
            distance = {}
            parent = {}
            queue = []
            for anchor in anchors:
                anchor = int(anchor)
                distance[anchor] = 0.0
                parent[anchor] = None
                heapq.heappush(queue, (0.0, anchor))
            while queue:
                cost, u = heapq.heappop(queue)
                if cost > distance.get(u, float('inf')) + 1e-12:
                    continue
                if u == int(target):
                    break
                if u in tree_nodes and u not in anchors:
                    continue
                for v, weight in adjacency.get(u, ()):
                    if v in tree_nodes or v in blocked_terminals:
                        continue
                    next_cost = cost + weight
                    if next_cost + 1e-12 >= distance.get(v, float('inf')):
                        continue
                    distance[v] = next_cost
                    parent[v] = u
                    heapq.heappush(queue, (next_cost, v))
            if int(target) not in parent:
                return None
            path = [int(target)]
            while parent[path[-1]] is not None:
                path.append(parent[path[-1]])
            path.reverse()
            return path

        states = [{
            'tree': occupied,
            'full': {root},
            'connected': set(),
            'remaining': pending,
            'cost': 0,
        }]
        for _ in range(len(pending)):
            expanded = []
            for state in states:
                for target in state['remaining']:
                    path = find_attachment(
                        target,
                        state['remaining'],
                        state['tree'],
                        state['full'],
                        state['connected'],
                    )
                    if path is None:
                        continue
                    path_nodes = set(path)
                    expanded.append({
                        'tree': state['tree'] | path_nodes,
                        'full': state['full'] | path_nodes,
                        'connected': state['connected'] | {int(target)},
                        'remaining': tuple(
                            node for node in state['remaining']
                            if int(node) != int(target)
                        ),
                        'cost': state['cost'] + max(0, len(path) - 1),
                    })
            if not expanded:
                return False
            expanded.sort(key=lambda item: (item['cost'], item['remaining']))
            states = expanded[:max(1, int(beam_width))]
        return bool(states)

    def _can_complete_after_placement(
            self, current_idx, candidate, bw_req, destinations,
            previous_override=None, prefix_occupied=None):
        """Bounded topology lookahead for the remaining VNF chain and branches.

        The request chains used by this project are short, so exploring the
        remaining DC stages is inexpensive and avoids selecting a locally valid
        placement that leaves no legal next stage or post-chain destination.
        """
        request = self.env.current_request or {}
        if not hasattr(self, '_completion_path_evidence'):
            self._completion_path_evidence = {}
        if not hasattr(self, '_completion_plan_evidence'):
            self._completion_plan_evidence = {}
        vnfs = list(request.get('vnf', []) or [])
        if not vnfs or current_idx >= len(vnfs):
            return True
        chain = list(getattr(self.env, 'chain_nodes', []) or [])
        source = request.get('source')
        previous = (
            previous_override
            if previous_override is not None
            else (chain[-1] if chain else source)
        )
        first_path = self._find_ordered_stage_path(
            previous, candidate, bw_req, blocked_nodes=destinations
        )
        if not first_path:
            return False

        existing_edges = self.shared.get_positive_tree_edge_set()
        occupied = {int(node) for edge in existing_edges for node in edge}
        occupied.update(int(node) for node in (prefix_occupied or ()))
        occupied.update(int(node) for node in first_path[:-1])
        cpu_list = request.get('cpu_origin', []) or request.get('vnf_cpu', [])
        mem_list = request.get('memory_origin', []) or request.get('vnf_mem', [])
        dc_nodes = tuple(int(node) for node in getattr(self.env, 'dc_nodes', ()))
        memo = {}

        def reserve_stage(node, stage_idx, reserved_cpu, reserved_mem,
                          planned_instances):
            """Apply one hypothetical VNF deployment to a planning ledger."""
            node = int(node)
            vnf_type = int(vnfs[stage_idx])
            req_cpu = (
                float(cpu_list[stage_idx])
                if stage_idx < len(cpu_list) else 10.0
            )
            req_mem = (
                float(mem_list[stage_idx])
                if stage_idx < len(mem_list) else 10.0
            )
            instance_key = (node, vnf_type)

            # A fresh instance selected earlier in this same hypothetical plan
            # becomes reusable by a repeated stage when it is really executed.
            if instance_key in planned_instances:
                return (
                    dict(reserved_cpu), dict(reserved_mem),
                    set(planned_instances),
                )

            probe = self.env.resource_mgr.probe_vnf_deploy(
                node, vnf_type, req_cpu, req_mem
            )
            if not probe.get('ok', False):
                return None
            if probe.get('reuse', False):
                return (
                    dict(reserved_cpu), dict(reserved_mem),
                    set(planned_instances),
                )

            cpu_left = float(
                self.env.resource_mgr.pool.get_available_cpu(node)
            ) - float(reserved_cpu.get(node, 0.0))
            mem_left = float(
                self.env.resource_mgr.pool.get_available_memory(node)
            ) - float(reserved_mem.get(node, 0.0))
            if cpu_left + 1e-9 < req_cpu or mem_left + 1e-9 < req_mem:
                return None
            next_cpu = dict(reserved_cpu)
            next_mem = dict(reserved_mem)
            next_instances = set(planned_instances)
            next_cpu[node] = float(next_cpu.get(node, 0.0)) + req_cpu
            next_mem[node] = float(next_mem.get(node, 0.0)) + req_mem
            next_instances.add(instance_key)
            return next_cpu, next_mem, next_instances

        def search(stage_idx, start, blocked_spine, reserved_cpu,
                   reserved_mem, planned_instances):
            key = (
                int(stage_idx), int(start), tuple(sorted(blocked_spine)),
                tuple(sorted(reserved_cpu.items())),
                tuple(sorted(reserved_mem.items())),
                tuple(sorted(planned_instances)),
            )
            if key in memo:
                return memo[key]
            if stage_idx >= len(vnfs):
                blocked = set(blocked_spine)
                blocked.discard(int(start))
                destination_ok = self._can_complete_destination_tree(
                    start,
                    destinations,
                    bw_req,
                    blocked,
                )
                result = [] if destination_ok else None
                memo[key] = result
                return result

            # Once a pre-final VNF is placed on a destination switch, all
            # remaining stages must be co-located there.
            candidates = (int(start),) if int(start) in destinations else dc_nodes
            for node in candidates:
                # A future VNF cannot be placed on an internal node already
                # occupied by the hypothetical SFC spine.  The generic path
                # finder unblocks its target to support destination co-location;
                # without this explicit gate it produced plans that necessarily
                # re-entered the physical tree during real execution.
                if int(node) in blocked_spine and int(node) != int(start):
                    continue
                reservation = reserve_stage(
                    node, stage_idx, reserved_cpu, reserved_mem,
                    planned_instances,
                )
                if reservation is None:
                    continue
                path = self._find_ordered_stage_path(
                    start,
                    node,
                    bw_req,
                    blocked_nodes=set(destinations).union(blocked_spine),
                )
                if not path:
                    continue
                next_blocked = set(blocked_spine)
                next_blocked.update(int(value) for value in path[:-1])
                suffix = search(
                    stage_idx + 1, node, next_blocked, *reservation
                )
                if suffix is not None:
                    result = [{
                        'stage_idx': int(stage_idx),
                        'node': int(node),
                        'path': [int(value) for value in path],
                    }] + list(suffix)
                    memo[key] = result
                    return result
            memo[key] = None
            return None

        initial_reservation = reserve_stage(
            int(candidate), int(current_idx), {}, {}, set()
        )
        if initial_reservation is None:
            return False
        suffix_plan = search(
            current_idx + 1,
            int(candidate),
            occupied,
            *initial_reservation,
        )
        result = suffix_plan is not None
        if result:
            self._completion_path_evidence[int(candidate)] = [
                int(node) for node in first_path
            ]
            self._completion_plan_evidence[int(candidate)] = [{
                'stage_idx': int(current_idx),
                'node': int(candidate),
                'path': [int(node) for node in first_path],
            }] + list(suffix_plan)
        return result

    def _ensure_static_topology_cache(self):
        if self._static_topology_ready:
            return
        n = int(getattr(self.env, 'n', 0))
        edges = []
        rm = getattr(self.env, 'resource_mgr', None)
        for u in range(n):
            if rm is not None and hasattr(rm, 'get_neighbors'):
                try:
                    neighbors = rm.get_neighbors(u)
                except Exception:
                    neighbors = []
            else:
                topology = getattr(self.env, 'topology', None)
                neighbors = [v for v in range(n) if topology is not None and v != u and topology[u][v] > 0]
            for v in neighbors:
                try:
                    v = int(v)
                    if 0 <= v < n:
                        edges.append((u, v))
                except Exception:
                    continue
        self._static_edge_uv = edges
        self._static_edge_index = (
            torch.tensor(edges, dtype=torch.long).t().contiguous()
            if edges else torch.zeros((2, 0), dtype=torch.long)
        )

        topology = getattr(self.env, 'topology', None)
        try:
            max_weight = max(1.0, float(np.asarray(topology).max()))
        except Exception:
            max_weight = 1.0
        self._static_edge_hop_weight = [
            float(topology[u, v]) / max_weight if topology is not None else 1.0
            for u, v in edges
        ]
        self._static_topology_ready = True




    def set_high_level_goal(self, high_action_idx, target_node_id, start_node_id=None):
        self.env.last_high_action_idx = high_action_idx
        target_node_id = int(target_node_id)

        # ============================================================
        
        # ============================================================
        if self.env.current_request:
            vnf_list = self.env.current_request.get('vnf', [])
            current_vnf_idx = getattr(self.env, 'next_vnf_idx', 0)
            if current_vnf_idx < len(vnf_list):
                dc_nodes = getattr(self.env, 'dc_nodes', set())
                _current_vnf_type = vnf_list[current_vnf_idx]
                _cpu_list = self.env.current_request.get('cpu_origin', []) or \
                            self.env.current_request.get('vnf_cpu', [])
                _mem_list = self.env.current_request.get('memory_origin', []) or \
                            self.env.current_request.get('vnf_mem', [])
                _req_cpu = float(_cpu_list[current_vnf_idx]) if current_vnf_idx < len(_cpu_list) else 10.0
                _req_mem = float(_mem_list[current_vnf_idx]) if current_vnf_idx < len(_mem_list) else 10.0
                _nodes_on_tree = set(getattr(self.env, 'nodes_on_tree', set()))
                
                _rm_sg = self.env.resource_mgr
                _probe_sg = _rm_sg.probe_vnf_deploy(
                    target_node_id, _current_vnf_type, _req_cpu, _req_mem
                )
                _resource_ok = _probe_sg['ok']
                _can_reuse   = _probe_sg['reuse']
                _is_dc_ok = (not dc_nodes) or (target_node_id in dc_nodes)
                if (not _is_dc_ok) or (not _resource_ok):
                    logger.warning(
                        f"[High.set_goal] 淇濈暀鍘熷姩浣滐紝涓嶅啀 silent replace | "
                        f"target={target_node_id} dc_ok={int(_is_dc_ok)} resource_ok={int(_resource_ok)} "
                        f"req_cpu={_req_cpu:.1f} req_mem={_req_mem:.1f} "
                        f"avail_cpu={self.env.resource_mgr.pool.get_available_cpu(target_node_id):.1f} "
                        f"avail_mem={self.env.resource_mgr.pool.get_available_memory(target_node_id):.1f} "
                        f"can_reuse={int(_can_reuse)}"
                    )

        self.env.current_subgoal_node = target_node_id
        self.env.subgoal_step_count = 0

        actual_location = self.env.current_node_location
        if start_node_id is not None:
            start_node_id = int(start_node_id)
            if start_node_id != actual_location:
                
                
                
                
                _phase = getattr(self.env, 'current_phase', None)
                _chain = getattr(self.env, 'chain_nodes', [])
                _anchor = getattr(self.env, 'current_anchor_node', None)
                
                _valid_start = None
                if _phase == 'destination_connection':
                    if _anchor is not None and start_node_id == _anchor:
                        _valid_start = _anchor
                    elif _chain and start_node_id == _chain[-1]:
                        _valid_start = _chain[-1]
                if _valid_start is not None:
                    self.env.current_node_location = start_node_id
                    logger.debug(
                        f"[High] valid destination-stage start switch: "
                        f"{actual_location}->{start_node_id} (anchor={_anchor})"
                    )
                else:
                    logger.warning(
                        f"[High] predicted start {start_node_id} != actual location "
                        f"{actual_location}; keep the actual location"
                    )
                
            else:
                logger.debug(f" [High] Agent璧风偣纭: {start_node_id}")

        if self.env.current_request:
            vnf_list = self.env.current_request.get('vnf', [])
            self.env.current_vnf_to_deploy = self.env.next_vnf_idx

            if self.env.next_vnf_idx < len(vnf_list):
                self.env.current_phase = 'vnf_deployment'
                self.env.current_deployment_target = target_node_id
                self.env.current_target_node = None
            else:
                self.env.current_phase = 'destination_connection'
                self.env.current_target_node = target_node_id
                self.env.current_deployment_target = None

        # Keep the complete executed path for structural SFT evidence. The
        # short current_path_trace remains only a tabu window.
        self.env.current_subgoal_full_path = [
            int(self.env.current_node_location)
        ]

        return self.get_high_level_state_graph()

    def step_high_level(self, action_idx):
        # ================================================================
        
        # ================================================================
        if self.env.current_request:
            vnf_list = self.env.current_request.get('vnf', [])
            current_vnf_idx = getattr(self.env, 'next_vnf_idx', 0)
            
            if current_vnf_idx >= len(vnf_list):
                if getattr(self.env, 'current_phase', None) == 'vnf_deployment':
                    self.env.current_phase = 'destination_connection'
                    self.env.current_deployment_target = None
                    logger.debug(f" [High.step] 妫€娴嬪埌VNF宸插叏瀹屾垚锛宲hase寮哄埗鍒囨崲鈫抎estination_connection")

        mask = self.get_high_level_action_mask()
        if np.sum(mask) == 0:
            phase = getattr(self.env, 'current_phase', 'unknown')
            vnf_idx = getattr(self.env, 'next_vnf_idx', -1)
            pending = []
            if self.env.current_request:
                all_d = set(self.env.current_request.get('dest', []))
                conn_d = self.shared.get_connected_dests_view()
                pending = list(all_d - conn_d)
            logger.warning(
                f" [High] 鏃犲彲琛岄珮灞傚姩浣?| phase={phase} | vnf_idx={vnf_idx} | pending_dests={pending}"
            )
            return None, -10.0, True, False, {'no_valid_action': True, 'phase': phase}

        
        try:
            action_idx = int(action_idx)
        except Exception:
            logger.warning(f" [High] 闈炴硶楂樺眰鍔ㄤ綔锛氭棤娉曡浆鎴恑nt | action={action_idx}")
            return None, -8.0, False, True, {
                'illegal_action': True,
                'reason': 'non_int_action'
            }

        if action_idx < 0 or action_idx >= len(mask) or mask[action_idx] <= 0:
            logger.warning(
                f" [High] 闈炴硶楂樺眰鍔ㄤ綔琚嫆缁?| action={action_idx} "
                f"in_range={int(0 <= action_idx < len(mask))} "
                f"mask_alive={int(0 <= action_idx < len(mask) and mask[action_idx] > 0)}"
            )
            return None, -8.0, False, True, {
                'illegal_action': True,
                'reason': 'masked_or_out_of_range',
                'action': action_idx
            }

        
        
        
        
        target_node = int(action_idx)
        self.current_high_action = target_node

        connected = set()
        if hasattr(self.env, 'current_tree') and self.env.current_tree:
            try:
                connected = {int(x) for x in self.shared.get_connected_dests_view()}
            except:
                pass

        all_dests = set()
        if self.env.current_request:
            try:
                all_dests = {int(x) for x in self.env.current_request.get('dest', [])}
            except:
                pass

        if target_node in connected:
            if all_dests.issubset(connected):
                logger.debug("[High] all destinations are connected; episode can finish")
                reward = 20.0
                if hasattr(self.env, 'low_level_controller') and hasattr(self.env.low_level_controller,
                                                                         '_calculate_tree_metrics'):
                    metrics = self.env.low_level_controller._calculate_tree_metrics()
                    # [Fix-1] Fix terminal reward direction
                    # Old: -3.0*redundancy penalized sharing efficiency (direction inverted)
                    #      -1.5*tree_n_edges could make successful episode score negative
                    # New: reward sharing rate; penalize only extra edges beyond minimum tree
                    _sharing_ratio = metrics.get('redundancy', 0.0)  # reused/total, higher=better
                    _tree_n = metrics.get('tree_n_edges', 0)
                    _n_dests = len(all_dests) if all_dests else 1
                    _min_tree = max(_n_dests, 1)   # minimum tree ~= number of destinations
                    _extra_edges = max(0, _tree_n - _min_tree)
                    reward += +2.0 * _sharing_ratio   # reward sharing, consistent with +2.5*delta_reuse
                    reward += -0.3 * _extra_edges      # only penalize truly redundant edges
                    logger.debug(
                        f"[TerminalReward] base=20.0 sharing={_sharing_ratio:.3f}(+{2.0*_sharing_ratio:.2f}) "
                        f"extra_edges={_extra_edges}(-{0.3*_extra_edges:.2f}) final={reward:.2f}"
                    )
                return None, reward, True, False, {'all_done': True}
            else:
                logger.warning(f"[High] target {target_node} is already completed; truncate with penalty")
                return None, -3.0, False, True, {
                    'subgoal_completed': True,
                    'warning': 'selected_completed_target'
                }

        step_penalty = -0.1

        return None, step_penalty, False, False, {
            'target_node': target_node,
            'status': 'executing'
        }

    
    
    

    def get_high_level_action_mask(self):
        n = self.env.n
        mask = np.zeros(n, dtype=np.float32)

        if not hasattr(self.env, 'current_request') or self.env.current_request is None:
            return np.ones(n, dtype=np.float32)

        # A coordinator cycle queries the same hard mask three times: once to
        # choose the action, once to build candidate features, and once in
        # step_high_level() for validation.  No tree or ledger mutation occurs
        # between those calls.  Reuse the coordinator-owned copy inside this
        # cycle; run_high_low_cycle() clears it before every new decision state.
        cycle_override = getattr(self.env, '_cycle_high_mask_override', None)
        if cycle_override is not None:
            cycle_override = np.asarray(cycle_override, dtype=np.float32)
            if cycle_override.shape == (n,):
                return cycle_override.copy()

        # A previously decoded complete VNF plan can provide one freshly
        # revalidated stage. Consume this one-shot override so the action check
        # in step_high_level() uses the same certificate as the coordinator.
        scheduled_override = getattr(
            self.env, '_scheduled_high_mask_override', None
        )
        if (
            getattr(self.env, 'current_phase', None) == 'vnf_deployment'
            and scheduled_override is not None
        ):
            scheduled_override = int(scheduled_override)
            if 0 <= scheduled_override < n:
                mask[scheduled_override] = 1.0
                return mask

        vnf_list = self.env.current_request.get('vnf', [])
        current_vnf_idx = getattr(self.env, 'next_vnf_idx', 0)

        
        if current_vnf_idx < len(vnf_list):
            if getattr(self.env, '_candidate_ablation', 'none') in ('no_ac', 'no_high_ac'):
                mask = np.ones(n, dtype=np.float32)
                if np.random.rand() < 0.01:
                    logger.info(
                        f"[CandidateAblation-High] mode={getattr(self.env, '_candidate_ablation', 'none')} "
                        f"phase=vnf_deployment open_nodes={int(np.sum(mask > 0))}/{n}"
                    )
                return mask

            cpu_list = self.env.current_request.get('cpu_origin', []) or \
                       self.env.current_request.get('vnf_cpu', [])
            mem_list = self.env.current_request.get('memory_origin', []) or \
                       self.env.current_request.get('vnf_mem', [])
            req_cpu = float(cpu_list[current_vnf_idx]) if current_vnf_idx < len(cpu_list) else 10.0
            req_mem = float(mem_list[current_vnf_idx]) if current_vnf_idx < len(mem_list) else 10.0

            
            
            
            
            _source = self.env.current_request.get('source', -1) if self.env.current_request else -1
            _dests = set(int(d) for d in self.env.current_request.get('dest', [])) if self.env.current_request else set()
            
            
            
            
            
            _placement = self.env.current_tree.get('placement', {}) if self.env.current_tree else {}
            _already = {key[0] for key in _placement
                        if isinstance(key, tuple) and len(key) >= 2
                        and key[1] == current_vnf_idx}
            _failed_deploy_nodes = set(getattr(self.env, '_episode_deploy_failed', set()) or set())
            _mask_diag = {
                'total_dc': len(getattr(self.env, 'dc_nodes', [])),
                'source': 0,
                'dest_candidate': 0,
                'already': 0,
                'failed_deploy': 0,
                'probe_fail': {},
                'bw_in_fail': 0,
                'ordered_path_fail': 0,
                'completion_lookahead_fail': 0,
                'resource_ok': 0,
                'strict_ok': 0,
                'loose_ok': 0,
                'fallback_ok': 0,
                'completion_fallback_used': 0,
                'sample': [],
            }

            
            
            _current_vnf_type = vnf_list[current_vnf_idx]
            _node_vnf_types = {}  # node -> set of deployed vnf_types
            for (pnode, _), pinfo in _placement.items():
                _node_vnf_types.setdefault(pnode, set()).add(pinfo.get('vnf_type'))
            
            _chain_nodes_ordered = getattr(self.env, 'chain_nodes', [])

            # If an earlier VNF was placed on a destination switch, the rest
            # of the chain must stay on that same switch.  This represents a
            # local service pipeline followed by host delivery and avoids an
            # invalid leave-and-return path in the physical-tree projection.
            _pinned_destination_dc = None
            if (
                current_vnf_idx > 0
                and _chain_nodes_ordered
                and int(_chain_nodes_ordered[-1]) in _dests
            ):
                _pinned_destination_dc = int(_chain_nodes_ordered[-1])

            def _can_colocate_remaining_chain(node):
                if node not in _dests:
                    return True
                if current_vnf_idx == len(vnf_list) - 1:
                    return True
                if _pinned_destination_dc is not None:
                    return int(node) == _pinned_destination_dc

                remaining_cpu = 0.0
                remaining_mem = 0.0
                for idx in range(current_vnf_idx, len(vnf_list)):
                    cpu = float(cpu_list[idx]) if idx < len(cpu_list) else 10.0
                    mem = float(mem_list[idx]) if idx < len(mem_list) else 10.0
                    probe = self.env.resource_mgr.probe_vnf_deploy(
                        node, vnf_list[idx], cpu, mem
                    )
                    if not probe.get('ok', False):
                        return False
                    if not probe.get('reuse', False):
                        remaining_cpu += cpu
                        remaining_mem += mem
                try:
                    return (
                        self.env.resource_mgr.pool.get_available_cpu(node) + 1e-9
                        >= remaining_cpu
                        and self.env.resource_mgr.pool.get_available_memory(node) + 1e-9
                        >= remaining_mem
                    )
                except (AttributeError, TypeError):
                    return True

            if hasattr(self.env, 'dc_nodes'):
                _llc = getattr(self.env, 'low_level_controller', None)
                
                
                
                
                
                _prev_vnf = (_chain_nodes_ordered[-1] if _chain_nodes_ordered
                             else (_source if _source != -1 else None))

                
                
                
                _can_use_order_constraint = (_llc is not None and _prev_vnf is not None and bool(_dests))

                if _can_use_order_constraint:
                    
                    _dist_prev_to_dests = {}
                    _nearest_dest_dist_prev = 9999
                    for _d in _dests:
                        _dd = _llc._get_hop_distance(_prev_vnf, _d)
                        _dist_prev_to_dests[_d] = _dd
                        if _dd < _nearest_dest_dist_prev:
                            _nearest_dest_dist_prev = _dd
                else:
                    _dist_prev_to_dests = {}
                    _nearest_dest_dist_prev = 9999

                _strict_mask = np.zeros(n, dtype=np.float32)
                _loose_mask  = np.zeros(n, dtype=np.float32)
                _cheap_feasible_mask = np.zeros(n, dtype=np.float32)
                # [Fix-2] \u4e3a VNF \u9636\u6bb5 BW \u5b64\u5c9b\u8fc7\u6ee4\u9884\u5907\u516c\u5171\u53d8\u91cf
                _pool = getattr(self.env.resource_mgr, 'pool', None)
                _bw_req_vnf = float(self.env.current_request.get('bw_origin', 0.0)) if self.env.current_request else 0.0
                _completion_budget = max(0, int(getattr(
                    self.env, '_online_completion_candidate_budget', 0
                ) or 0))
                _completion_enabled = self.env.config.get(
                    'high_completion_lookahead', True
                )
                self._completion_path_evidence = {}
                self._completion_plan_evidence = {}
                _completion_cache = {}

                def _completion_ok(node):
                    node = int(node)
                    if node not in _completion_cache:
                        _completion_cache[node] = self._can_complete_after_placement(
                            current_vnf_idx, node, _bw_req_vnf, _dests
                        )
                    return _completion_cache[node]

                
                
                
                _cur_loc = getattr(self.env, 'current_node_location', _source)
                _dist_cur_to_dc = {}
                if _llc is not None and _cur_loc is not None:
                    for _dc in self.env.dc_nodes:
                        _dist_cur_to_dc[_dc] = _llc._get_hop_distance(_cur_loc, _dc)
                    _min_dc_dist = min(_dist_cur_to_dc.values()) if _dist_cur_to_dc else 0
                    _max_dc_dist_allowed = max(9, _min_dc_dist * 3 + 1)  # [Relax-H1]
                else:
                    _min_dc_dist = 0
                    _max_dc_dist_allowed = 9999

                for node in self.env.dc_nodes:
                    if 0 <= node < n:
                        if (_pinned_destination_dc is not None
                                and node != _pinned_destination_dc):
                            continue
                        if node == _source:
                            _mask_diag['source'] += 1
                            continue
                        # A destination switch may host an earlier stage only
                        # when all remaining stages can be co-located there.
                        if node in _dests:
                            _mask_diag['dest_candidate'] += 1
                            if not _can_colocate_remaining_chain(node):
                                continue
                        if node in _already:
                            _mask_diag['already'] += 1
                            continue
                        if node in _failed_deploy_nodes:
                            _mask_diag['failed_deploy'] += 1
                            continue
                        _has_same_type = (_current_vnf_type in _node_vnf_types.get(node, set()))
                        _nodes_on_tree = set(getattr(self.env, 'nodes_on_tree', set()))

                        
                        _probe = self.env.resource_mgr.probe_vnf_deploy(
                            node, _current_vnf_type, req_cpu, req_mem
                        )
                        if not _probe['ok']:
                            _reason = str(_probe.get('reason', 'unknown'))
                            _mask_diag['probe_fail'][_reason] = _mask_diag['probe_fail'].get(_reason, 0) + 1
                            if len(_mask_diag['sample']) < 5:
                                try:
                                    _mask_diag['sample'].append(
                                        (int(node), _reason,
                                         round(float(self.env.resource_mgr.pool.get_available_cpu(node)), 1),
                                         round(float(self.env.resource_mgr.pool.get_available_memory(node)), 1))
                                    )
                                except Exception:
                                    _mask_diag['sample'].append((int(node), _reason))
                            logger.debug(
                                f"[Mask-Probe] 鑺傜偣{node} vnf={_current_vnf_type} "
                                f"涓嶅彲閮ㄧ讲 reason={_probe['reason']}"
                            )
                            continue
                        
                        
                        _mask_diag['resource_ok'] += 1
                        if _pool is not None and _bw_req_vnf > 0:
                            _nbrs_vnf = self.env.resource_mgr.get_neighbors(node)
                            _tree_edges_vnf = self.shared.get_positive_tree_edge_set()
                            _has_bw_vnf = any(
                                (_pool.get_available_bandwidth(_nb, node) >= _bw_req_vnf)
                                or ((_nb, node) in _tree_edges_vnf)
                                for _nb in _nbrs_vnf
                            )
                            if not _has_bw_vnf:
                                _mask_diag['bw_in_fail'] += 1
                                logger.debug(f"[Mask-VNF-BW] DC={node} all incoming BW insufficient; filtered")
                                continue
                        if not self._has_ordered_stage_path(
                                _prev_vnf, node, _bw_req_vnf, blocked_nodes=_dests):
                            _mask_diag['ordered_path_fail'] += 1
                            continue
                        if (
                            _completion_enabled
                            and _completion_budget <= 0
                            and not _completion_ok(node)
                        ):
                            _mask_diag['completion_lookahead_fail'] += 1
                            continue
                        _resource_ok = True
                        if _resource_ok:
                            _cheap_feasible_mask[node] = 1.0
                            if not _can_use_order_constraint:
                                
                                _loose_mask[node] = 1.0
                                continue

                            _prev_to_node = _llc._get_hop_distance(_prev_vnf, node)

                            
                            _nearest_dest_dist_node = 9999
                            for _d in _dests:
                                _dd = _llc._get_hop_distance(node, _d)
                                if _dd < _nearest_dest_dist_node:
                                    _nearest_dest_dist_node = _dd

                            
                            _on_shortest_path = False
                            for _d in _dests:
                                _d_prev = _dist_prev_to_dests.get(_d, 9999)
                                _d_node = _llc._get_hop_distance(node, _d)
                                if _prev_to_node + _d_node == _d_prev:
                                    _on_shortest_path = True
                                    break

                            
                            _dist_to_cur = _dist_cur_to_dc.get(node, 9999)
                            _too_far = (_dist_to_cur > _max_dc_dist_allowed)

                            # [Relax-H3] strict = closer to dest than prev_vnf
                            if _nearest_dest_dist_node < _nearest_dest_dist_prev:
                                _strict_mask[node] = 1.0
                                _mask_diag['strict_ok'] += 1
                            # loose = equal or closer distance
                            if _nearest_dest_dist_node <= _nearest_dest_dist_prev:
                                _loose_mask[node] = 1.0
                                _mask_diag['loose_ok'] += 1


                
                if np.sum(_strict_mask) > 0:
                    mask = _strict_mask
                elif np.sum(_loose_mask) > 0:
                    mask = _loose_mask
                else:
                    
                    for node in self.env.dc_nodes:
                        if (0 <= node < n and node != _source
                                and (_pinned_destination_dc is None
                                     or node == _pinned_destination_dc)
                                and _can_colocate_remaining_chain(node)
                                and node not in _already and node not in _failed_deploy_nodes):
                            _has_same_type = (_current_vnf_type in _node_vnf_types.get(node, set()))
                            _nodes_on_tree = set(getattr(self.env, 'nodes_on_tree', set()))
                            
                            _probe_fb = self.env.resource_mgr.probe_vnf_deploy(
                                node, _current_vnf_type, req_cpu, req_mem
                            )
                            if (
                                _probe_fb['ok']
                                and self._has_ordered_stage_path(
                                    _prev_vnf, node, _bw_req_vnf,
                                    blocked_nodes=_dests
                                )
                                and (
                                    not _completion_enabled
                                    or _completion_budget > 0
                                    or _completion_ok(node)
                                )
                            ):
                                mask[node] = 1.0
                                _mask_diag['fallback_ok'] += 1

                if (
                    _completion_enabled
                    and _completion_budget > 0
                    and np.sum(mask) > 0
                ):
                    # The cheap mask above already enforces CPU, memory,
                    # bandwidth, SFC order, and destination co-location.  Rank
                    # only those candidates, then run the expensive recursive
                    # completion proof until enough fully feasible choices are
                    # available for the high-level RL policy.
                    def _online_completion_rank(node):
                        node = int(node)
                        probe = self.env.resource_mgr.probe_vnf_deploy(
                            node, _current_vnf_type, req_cpu, req_mem
                        )
                        nearest_dest = min(
                            (_llc._get_hop_distance(node, dest) for dest in _dests),
                            default=9999,
                        ) if _llc is not None else 9999
                        return (
                            0 if probe.get('reuse', False) else 1,
                            nearest_dest,
                            _dist_cur_to_dc.get(node, 9999),
                            node,
                        )

                    def _verify_candidate_tier(candidate_mask):
                        ranked_nodes = sorted(
                            (
                                int(node)
                                for node in np.where(candidate_mask > 0)[0]
                            ),
                            key=_online_completion_rank,
                        )
                        verified = np.zeros(n, dtype=np.float32)
                        for node in ranked_nodes:
                            if _completion_ok(node):
                                verified[node] = 1.0
                                if int(np.sum(verified)) >= _completion_budget:
                                    break
                            else:
                                _mask_diag['completion_lookahead_fail'] += 1
                        return verified

                    # Preserve the legacy fallback semantics.  A non-empty
                    # cheap strict tier can still contain zero candidates that
                    # admit a complete remaining SFT; in that case continue to
                    # the loose and general feasible tiers instead of returning
                    # a false all-zero mask.
                    tier_masks = [mask]
                    if not np.array_equal(mask, _loose_mask):
                        tier_masks.append(_loose_mask)
                    if not np.array_equal(mask, _cheap_feasible_mask):
                        tier_masks.append(_cheap_feasible_mask)
                    verified_mask = np.zeros(n, dtype=np.float32)
                    for tier_mask in tier_masks:
                        if np.sum(tier_mask) <= 0:
                            continue
                        verified_mask = _verify_candidate_tier(tier_mask)
                        if np.sum(verified_mask) > 0:
                            break
                    if (
                        np.sum(verified_mask) <= 0
                        and np.sum(_cheap_feasible_mask) > 0
                    ):
                        # Completion lookahead is a conservative future
                        # feasibility certificate.  If it proves no full
                        # suffix within the bounded budget, retain a bounded
                        # set of candidates that already passed the local
                        # resource, bandwidth, and ordered-path checks.  The
                        # normal step validator and final SFC validator still
                        # reject any candidate whose remaining suffix cannot
                        # be completed; this avoids converting a proof miss
                        # into an unconditional false rejection.
                        fallback_nodes = sorted(
                            (
                                int(node)
                                for node in np.where(_cheap_feasible_mask > 0)[0]
                            ),
                            key=_online_completion_rank,
                        )[: max(1, _completion_budget)]
                        for node in fallback_nodes:
                            verified_mask[node] = 1.0
                        _mask_diag['completion_fallback_used'] = len(fallback_nodes)
                    mask = verified_mask
                # Last-resort safety tier: if route-order or conservative
                # completion lookahead removed every candidate, retain nodes
                # that still pass the authoritative CPU/MEM/reuse probe.  The
                # low-level validator remains responsible for proving the
                # actual path, so a proof miss no longer becomes a false
                # "all DC resources insufficient" rejection.
                if np.sum(mask) <= 0:
                    resource_only = np.zeros(n, dtype=np.float32)
                    for node in self.env.dc_nodes:
                        if not (0 <= int(node) < n):
                            continue
                        if int(node) == int(_source):
                            continue
                        if (_pinned_destination_dc is not None
                                and int(node) != int(_pinned_destination_dc)):
                            continue
                        if int(node) in _already or int(node) in _failed_deploy_nodes:
                            continue
                        try:
                            probe_resource = self.env.resource_mgr.probe_vnf_deploy(
                                int(node), _current_vnf_type, req_cpu, req_mem
                            )
                            if probe_resource.get('ok', False):
                                resource_only[int(node)] = 1.0
                        except Exception:
                            continue
                    if np.sum(resource_only) > 0:
                        mask = resource_only
                        _mask_diag['resource_only_fallback'] = int(np.sum(resource_only))
            if np.sum(mask) == 0:
                logger.warning(
                    f"[MaskZero-VNF] no feasible DC candidates | "
                    f"vnf_idx={current_vnf_idx}/{len(vnf_list)} "
                    f"vnf_type={_current_vnf_type} req_cpu={req_cpu:.1f} req_mem={req_mem:.1f} "
                    f"source={_source} dests={sorted(_dests)} already={sorted(_already)} "
                    f"failed_deploy={sorted(_failed_deploy_nodes)} diag={_mask_diag}"
                )
                logger.warning(
                    " [Mask] no feasible high-level DC action; "
                    "resource/path/order/completion filters eliminated all candidates"
                )
                return np.zeros(n, dtype=np.float32)
            else:
                open_dc = [i for i in range(n) if mask[i] > 0]
                logger.debug(f"[Mask] VNF闃舵寮€鏀綝C鑺傜偣: {open_dc}")

        
        else:
            try:
                all_dests = {int(x) for x in self.env.current_request.get('dest', [])}
                connected_set = set()
                if hasattr(self.env, 'current_tree') and self.env.current_tree:
                    connected_set = {int(x) for x in self.shared.get_connected_dests_view()}
            except:
                all_dests, connected_set = set(), set()

            true_pending = list(all_dests - connected_set)
            if not true_pending:
                return np.zeros(n, dtype=np.float32)

            
            llc = getattr(self.env, 'low_level_controller', None)
            last_vnf = self.env.chain_nodes[-1] if getattr(self.env, 'chain_nodes', []) else None
            _bw_req = float(self.env.current_request.get('bw_origin', 0.0)) if self.env.current_request else 0.0
            _pool = getattr(self.env.resource_mgr, 'pool', None)

            
            
            try:
                import networkx as nx
                _tree_edges_mask = self.shared.get_positive_tree_edge_set()
                _G_bw_mask = nx.DiGraph()
                for _u in range(n):
                    if not hasattr(self.env.resource_mgr, 'get_neighbors'):
                        break
                    for _v in self.env.resource_mgr.get_neighbors(_u):
                        _avail = _pool.get_available_bandwidth(_u, _v) if _pool else 0.0
                        if _avail < _bw_req and (_u, _v) not in _tree_edges_mask:
                            continue
                        _mask_diag['resource_ok'] += 1
                        _G_bw_mask.add_edge(_u, _v)
                _bw_graph_ok = True
            except Exception:
                _G_bw_mask = None
                _bw_graph_ok = False

            
            
            _tree_nodes_mask = set()
            if self.env.current_tree:
                for ek, fl in self.env.current_tree.get('tree', {}).items():
                    try:
                        if float(fl) > 0.0:
                            _tree_nodes_mask.add(int(ek[0]))
                            _tree_nodes_mask.add(int(ek[1]))
                    except Exception:
                        pass
            
            if not _tree_nodes_mask:
                _anchor_now = getattr(self.env, 'current_anchor_node', None)
                _fallback = _anchor_now if _anchor_now is not None else last_vnf
                if _fallback is not None:
                    _tree_nodes_mask = {_fallback}

            for node in true_pending:
                if 0 <= node < n:
                    
                    if llc is not None and last_vnf is not None:
                        if llc._get_hop_distance(last_vnf, node) >= 9999:
                            continue
                    if _pool is not None and _bw_req > 0:
                        _neighbors = self.env.resource_mgr.get_neighbors(node)
                        _has_bw = any(
                            (_pool.get_available_bandwidth(_nb, node) >= _bw_req)
                            or ((_nb, node) in _tree_edges_mask)
                            for _nb in _neighbors
                        )
                        if not _has_bw:
                            logger.debug(f"[Mask] dest={node} has no feasible inbound bandwidth; masked")
                            continue
                    
                    
                    if (_bw_graph_ok and _G_bw_mask is not None
                            and _tree_nodes_mask and _bw_req > 0):
                        
                        _total_vnf_m = len(self.env.current_request.get('vnf', [])) \
                            if self.env.current_request else 0
                        _node_stage_m = self.env.current_tree.get('node_stage', {}) \
                            if self.env.current_tree else {}
                        if _total_vnf_m > 0:
                            _anchor_candidates = {
                                n for n in _tree_nodes_mask
                                if _node_stage_m.get(n, 0) >= _total_vnf_m
                            } or _tree_nodes_mask
                        else:
                            _anchor_candidates = _tree_nodes_mask
                        _dest_in = node in _G_bw_mask
                        if not _dest_in:
                            continue
                        
                        
                        _cur_anchor_m = getattr(self.env, 'current_anchor_node', None)
                        _reachable = False
                        _cur_anchor_reachable = False
                        _reachable_anchors = []
                        if _cur_anchor_m is not None and _cur_anchor_m in _G_bw_mask:
                            _cur_anchor_reachable = nx.has_path(_G_bw_mask, _cur_anchor_m, node)
                            _reachable = _cur_anchor_reachable
                        if not _reachable:
                            for anchor in _anchor_candidates:
                                if anchor in _G_bw_mask and nx.has_path(_G_bw_mask, anchor, node):
                                    _reachable_anchors.append(anchor)
                            _reachable = bool(_reachable_anchors)
                            if _reachable and _cur_anchor_m is not None:
                                logger.info(
                                    f"[DestMaskAnchorFallback] dest={node} "
                                    f"cur_anchor={_cur_anchor_m} unreachable, "
                                    f"fallback_anchors={sorted(_reachable_anchors)[:8]} "
                                    f"total_fallback={len(_reachable_anchors)}"
                                )
                        if not _reachable:
                            logger.info(
                                f"[Mask-BW] dest={node} full_stage鑺傜偣鍧嘊W璺緞涓嶅彲杈撅紝灞忚斀 "
                                f"cur_anchor={_cur_anchor_m} "
                                f"cur_anchor_reachable={int(_cur_anchor_reachable)} "
                                f"anchors={sorted(_anchor_candidates)} "
                                f"full_stage_cnt={len(_anchor_candidates)} "
                                f"bw_req={_bw_req}"
                            )
                            continue
                    mask[node] = 1.0

            
            
            
            
            
            
            if np.sum(mask) == 0 and true_pending:
                logger.debug(
                    f"[Mask-BW] BW璺緞杩囨护鍚巑ask鍏?锛屾棤鍙揪鐨刣est锛屼笉鍥為€€ pending={true_pending}"
                )
                
                return np.zeros(n, dtype=np.float32)


            for done_node in connected_set:
                if 0 <= done_node < n:
                    mask[done_node] = 0.0

        return mask

    def get_high_level_candidates(self):
        """
        鍦ㄩ珮灞?action mask 鍩虹涓婃瀯閫犵粨鏋勫寲鍊欓€夌壒寰併€?
        鍏蜂綋鐗瑰緛鏋勯€犲凡涓嬫矇鍒?ControllerSharedHelper銆?
        """
        mask = self.get_high_level_action_mask()
        return self.shared.build_high_level_candidates(
            mask=mask,
            hop_distance_fn=self.shared.get_hop_distance_lazy,
        )

    
    
    
    

    def get_high_level_state_graph(self):
        n = self.env.n
        
        
        
        _node_feat_dim = 11
        if hasattr(self.env, 'config'):
            _node_feat_dim = self.env.config.get('gnn', {}).get('node_feat_dim', 11)

        if not self.env.current_request:
            return Data(
                x=torch.zeros((n, _node_feat_dim), dtype=torch.float32),
                edge_index=torch.zeros((2, 0), dtype=torch.long),
                edge_attr=torch.zeros((0, 5), dtype=torch.float32),
                global_attr=torch.zeros((1, 5), dtype=torch.float32)
            )

        req = self.env.current_request
        vnf_list = req.get('vnf', [])
        source = req.get('source')
        dests = req.get('dest', [])

        connected_dests = self.shared.get_connected_dests_view()
        nodes_on_tree = getattr(self.env, 'nodes_on_tree', set())

        next_vnf_idx = getattr(self.env, 'next_vnf_idx', 0)
        is_vnf_phase = next_vnf_idx < len(vnf_list)

        req_cpu, req_mem = 0.0, 0.0
        if is_vnf_phase:
            cpu_list = req.get('cpu_origin') or req.get('vnf_cpu', [])
            mem_list = req.get('memory_origin') or req.get('vnf_mem', [])
            if next_vnf_idx < len(cpu_list):
                req_cpu = float(cpu_list[next_vnf_idx])
                req_mem = float(mem_list[next_vnf_idx]) if next_vnf_idx < len(mem_list) else 1.0

        node_vnf_counts = [0] * n
        placement = self.env.current_tree.get('placement', {})
        for placement_key, info in placement.items():
            node_id = None
            if isinstance(placement_key, tuple) and len(placement_key) >= 1:
                node_id = placement_key[0]
            elif isinstance(info, dict):
                node_id = info.get('node')
            if node_id is not None and 0 <= node_id < n:
                node_vnf_counts[node_id] += 1

        if hasattr(self.env, 'resource_mgr') and hasattr(self.env.resource_mgr, 'hvt_all'):
            hvt = self.env.resource_mgr.hvt_all
            for node_id in range(n):
                global_count = int(np.sum(hvt[node_id]))
                node_vnf_counts[node_id] = max(node_vnf_counts[node_id], global_count)

        dc_nodes = getattr(self.env, 'dc_nodes', set())

        
        _pending_d = [d for d in dests if d not in connected_dests]
        _use_reach = not getattr(self.env, '_ablation_reach', False)
        try:
            _reach = self.env.resource_mgr.get_reach_feats(_pending_d) if _use_reach else None
        except Exception as _e:
            logger.warning(f"[Reach] high-level reachability feature failed; using zeros: {_e}")
            _reach = None

        x = []
        for node in range(n):
            try:
                avail_cpu = self.env.resource_mgr.pool.get_available_cpu(node)
                avail_mem = self.env.resource_mgr.pool.get_available_memory(node)
                
                
                _c_cap = max(1.0, float(getattr(self.env.resource_mgr, 'C_cap', 100.0)))
                _m_cap = max(1.0, float(getattr(self.env.resource_mgr, 'M_cap', 100.0)))
                norm_cpu = min(avail_cpu / _c_cap, 1.0)
                norm_mem = min(avail_mem / _m_cap, 1.0)
            except:
                norm_cpu, norm_mem = 0.5, 0.5
                avail_cpu, avail_mem = 50.0, 50.0
                _c_cap, _m_cap = 100.0, 100.0

            features = [norm_cpu, norm_mem]
            features.append(1.0 if node == source else 0.0)

            is_dest = node in dests
            is_connected = node in connected_dests
            if is_dest:
                if is_connected:
                    is_pending = -1.0
                else:
                    is_pending = 0.5 if is_vnf_phase else 2.0
            else:
                is_pending = 0.0
            features.append(is_pending)

            features.append(1.0 if is_connected else 0.0)
            features.append(1.0 if node in nodes_on_tree else 0.0)

            if is_vnf_phase and req_cpu > 0:
                cpu_match = min(avail_cpu / max(req_cpu, 0.1), 1.0)
                mem_match = min(avail_mem / max(req_mem, 0.1), 1.0)
                match_score = 0.7 * cpu_match + 0.3 * mem_match
                if node in dc_nodes:
                    match_score = min(match_score + 0.1, 1.0)
            else:
                match_score = 0.0
            features.append(match_score)

            try:
                degree = len(self.env.resource_mgr.get_neighbors(node))
                norm_degree = min(degree / 10.0, 1.0)
            except:
                norm_degree = 0.5
            features.append(norm_degree)

            features.append(min(node_vnf_counts[node] / 5.0, 1.0))
            features.append(0.0)

            if is_vnf_phase:
                if node in dc_nodes and avail_cpu >= req_cpu and avail_mem >= req_mem:
                    load_penalty = node_vnf_counts[node] * 0.3
                    phase_guide = max(1.5 - load_penalty, 0.1)
                elif node in dc_nodes:
                    phase_guide = -1.0
                else:
                    phase_guide = -0.3
            else:
                if is_dest:
                    if is_connected:
                        phase_guide = -3.0
                    else:
                        phase_guide = 3.0
                elif node == source:
                    phase_guide = 0.5
                else:
                    pending_dests = [d for d in dests if d not in connected_dests]
                    if pending_dests:
                        
                        
                        _llc_ref = getattr(self.env, 'low_level_controller', None)
                        min_dist = float('inf')
                        for dest in pending_dests:
                            if _llc_ref is not None:
                                dist = _llc_ref._get_hop_distance(node, dest)
                            else:
                                dist = 9999
                            if dist < min_dist:
                                min_dist = dist
                        if min_dist == 0:
                            phase_guide = 2.5
                        elif min_dist == 1:
                            phase_guide = 1.5
                        elif min_dist <= 3:
                            phase_guide = 0.5
                        elif min_dist <= 6:
                            phase_guide = 0.0
                        else:
                            phase_guide = -0.5
                    else:
                        phase_guide = -1.0
            features.append(phase_guide)
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            features.append(1.0 if node in dc_nodes else 0.0)         # dim11: is_dc
            cur_loc = getattr(self.env, 'current_node_location', -1)
            features.append(1.0 if node == cur_loc else 0.0)           # dim12: is_current
            _subgoal = getattr(self.env, 'current_subgoal_node', None)
            if _subgoal is not None:
                _llc_ref2 = getattr(self.env, 'low_level_controller', None)
                _h = _llc_ref2._get_hop_distance(node, _subgoal) if _llc_ref2 else 9999
                features.append(min(_h / 10.0, 1.0))                   # dim13: hop_to_target
            else:
                features.append(0.5)
            
            if hasattr(self.env, 'resource_mgr') and hasattr(self.env.resource_mgr, 'hvt_all'):
                _hvt = self.env.resource_mgr.hvt_all[node]
                features.append(min(float(_hvt[0]) / 5.0, 1.0))       # dim14
                features.append(min(float(_hvt[1]) / 5.0, 1.0) if len(_hvt) > 1 else 0.0)  # dim15
                features.append(min(float(_hvt[2]) / 5.0, 1.0) if len(_hvt) > 2 else 0.0)  # dim16
            else:
                features.extend([0.0, 0.0, 0.0])
            
            features.append(1.0 if node == _subgoal else 0.0)          # dim17
            
            _vnf_total = max(1, len(vnf_list))
            features.append(float(next_vnf_idx) / _vnf_total)          # dim18
            
            _dest_total = max(1, len(dests))
            features.append(len(connected_dests) / _dest_total)        # dim19
            
            features.append(0.0 if is_vnf_phase else 1.0)              # dim20
            
            
            
            
            _feat21, _feat22 = 0.0, 0.0
            if is_vnf_phase and req_cpu > 0:
                try:
                    _current_vnf_type_f = vnf_list[next_vnf_idx] if next_vnf_idx < len(vnf_list) else -1
                    _probe_f = self.env.resource_mgr.probe_vnf_deploy(
                        node, _current_vnf_type_f, req_cpu, req_mem
                    )
                    _feat22 = 1.0 if _probe_f.get('ok', False) else 0.0
                    _feat21 = 1.0 if (_probe_f.get('ok', False) and _probe_f.get('reuse', False)) else 0.0
                except Exception:
                    pass
            features.extend([_feat21, _feat22, 0.0])                   # dim21-23
            
            if _reach is not None:
                features.extend(_reach[node].tolist())
            else:
                features.extend([0.0, 0.0, 0.0, 0.0])
            
            if len(features) < _node_feat_dim:
                features.extend([0.0] * (_node_feat_dim - len(features)))
            elif len(features) > _node_feat_dim:
                features = features[:_node_feat_dim]
            x.append(features)

        x_tensor = torch.tensor(x, dtype=torch.float32)

        self._ensure_static_topology_cache()
        edge_attr_list = []
        pool = getattr(getattr(self.env, 'resource_mgr', None), 'pool', None)
        for edge_idx, (u, v) in enumerate(self._static_edge_uv):
            try:
                if pool is not None:
                    cap = pool.bw_cap.get((u, v), 100.0)
                    available_bw = pool.get_available_bandwidth(u, v)
                    norm_bw = min(available_bw / max(1.0, cap), 1.0)
                    bw_util = 1.0 - norm_bw
                    hop_w = self._static_edge_hop_weight[edge_idx]
                    is_in_tree = 1.0 if self.shared.is_tree_edge(u, v) else 0.0
                    reserved = pool.bw_reserved.get((u, v), 0.0)
                    reserved_ratio = reserved / max(1.0, cap)
                    edge_attr_list.append([norm_bw, bw_util, hop_w, is_in_tree, reserved_ratio])
                else:
                    edge_attr_list.append([0.5, 0.5, 1.0, 0.0, 0.0])
            except Exception:
                edge_attr_list.append([0.5, 0.5, 1.0, 0.0, 0.0])
        edge_index = self._static_edge_index
        edge_attr = torch.tensor(edge_attr_list, dtype=torch.float32)

        bw_req = req.get('bw_origin', 0.0)
        norm_bw_req = min(bw_req / 10.0, 1.0)
        vnf_progress = next_vnf_idx / max(1, len(vnf_list))
        dest_progress = len(connected_dests) / max(1, len(dests))
        phase_feat = 0.0 if is_vnf_phase else 1.0
        
        
        
        
        _c_cap_g = max(1.0, float(getattr(self.env.resource_mgr, 'C_cap', 100.0)))
        _m_cap_g = max(1.0, float(getattr(self.env.resource_mgr, 'M_cap', 100.0)))
        _dc_nodes = getattr(self.env.resource_mgr, 'dc_nodes', list(range(n)))
        _n_dc = max(1, len(_dc_nodes))
        total_avail_cpu = sum(self.env.resource_mgr.pool.get_available_cpu(i) for i in _dc_nodes)
        total_avail_mem = sum(self.env.resource_mgr.pool.get_available_memory(i) for i in _dc_nodes)
        cpu_tension = 1.0 - (total_avail_cpu / (_n_dc * _c_cap_g))
        mem_tension = 1.0 - (total_avail_mem / (_n_dc * _m_cap_g))
        resource_tension = max(cpu_tension, mem_tension)

        global_attr = torch.tensor([[
            norm_bw_req, vnf_progress, dest_progress, phase_feat, resource_tension
        ]], dtype=torch.float32)

        return Data(x=x_tensor, edge_index=edge_index, edge_attr=edge_attr, global_attr=global_attr)

    
    
    
    


    
    
    

    def _is_all_tasks_completed(self):
        if not self.env.current_request:
            return True, "no_request"
        vnf_list = self.env.current_request.get('vnf', [])
        next_vnf_idx = getattr(self.env, 'next_vnf_idx', 0)
        vnf_done = next_vnf_idx >= len(vnf_list)
        dests = set(self.env.current_request.get('dest', []))
        connected = self.shared.get_connected_dests_view()
        dests_done = dests.issubset(connected)
        if vnf_done and dests_done:
            return True, "all_tasks_completed"
        elif vnf_done:
            return False, f"vnf_done_pending_dests:{len(dests) - len(connected)}"
        elif dests_done:
            return False, f"dests_done_pending_vnfs:{len(vnf_list) - next_vnf_idx}"
        else:
            return False, f"vnf:{next_vnf_idx}/{len(vnf_list)},dest:{len(connected)}/{len(dests)}"

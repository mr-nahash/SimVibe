# SimVibe – Hybrid Structural + Harmonic Vibrational Ligand Generator
# Hybrid structure + vibration scoring • Safe generation • Full documentation
# Uses a classical harmonic spring model + normal mode analysis (no quantum chemistry).
import logging
import math
import random
from io import StringIO
import numpy as np
import pandas as pd
import streamlit as st
from rdkit import Chem
from rdkit.Chem import AllChem, Draw, Descriptors, Crippen
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator
from rdkit.DataStructs import FingerprintSimilarity
from rdkit.Chem import rdChemReactions  # For simple reaction-based mutations
# ----------------------------- LOGGING CONFIG -----------------------------
logger = logging.getLogger("SimVibe")
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
logger.setLevel(logging.INFO)
# ----------------------------- STREAMLIT CONFIG -----------------------------
st.set_page_config(
    page_title="SimVibe Ligand Generator",
    page_icon="🧪",
    layout="wide",
)
morgan_gen = GetMorganGenerator(radius=2, fpSize=2048)
def get_morgan_fp(mol):
    return morgan_gen.GetFingerprint(mol)
# ----------------------------- ODOR PHYS-CHEM DESCRIPTOR (MW + logP) -----------------------------
def odor_physchem_score(
    smiles: str,
    mw_target: tuple[float, float] = (120.0, 200.0),
    logp_min: float = 1.0,
) -> tuple[float, float, float]:
    """
    Compute a simple 'odor-like' physchem score for small,
    hydrophobic, volatile/semi-volatile odorant-like ligands.
    - MW window ~ [mw_min, mw_max] Da (proxy for volatility / small size)
    - logP >= logp_min (hydrophobic or moderately hydrophobic)
    Returns:
        (odor_score_01, mw, logp)
        odor_score_01 is in [0,1].
    """
    mol = Chem.MolFromSmiles(smiles)
    if not mol:
        return 0.0, float("nan"), float("nan")
    mw = Descriptors.MolWt(mol)
    logp = Crippen.MolLogP(mol)
    # --- MW term: triangular membership inside [mw_min, mw_max] ---
    mw_min, mw_max = mw_target
    if mw <= mw_min or mw >= mw_max:
        mw_score = 0.0
    else:
        center = 0.5 * (mw_min + mw_max)
        half_range = 0.5 * (mw_max - mw_min)
        # 1.0 at center, 0 at edges
        mw_score = max(0.0, 1.0 - abs(mw - center) / half_range)
    # --- logP term: 0 below threshold, saturates to 1 at moderate hydrophobicity ---
    if logp <= logp_min:
        logp_score = 0.0
    else:
        # Simple linear ramp that saturates at logP ~ 5.0
        logp_score = min((logp - logp_min) / (5.0 - logp_min), 1.0)
    # Final odor physchem score = average of MW and logP scores
    odor_score = 0.5 * (mw_score + logp_score)
    return float(odor_score), float(mw), float(logp)
# ----------------------------- HARMONIC VIBRATIONAL DESCRIPTOR (CLASSICAL) -----------------------------
@st.cache_data(show_spinner=False)
def compute_vib_descriptor(smiles: str, max_modes: int = 25) -> np.ndarray:
    """
    Compute a vibrational descriptor using a classical harmonic spring model
    and mass-weighted normal mode analysis.
    1. SMILES -> RDKit Mol -> 3D geometry (UFF-optimized)
    2. Build a bonded "spring" network (each bond is a harmonic spring)
    3. Construct the mass-weighted Hessian (3N x 3N)
    4. Diagonalize to obtain normal mode eigenvalues
    5. Convert to "frequencies" (sqrt of eigenvalues), sort, truncate/pad to max_modes
]
    """
    mol = Chem.MolFromSmiles(smiles)
    if not mol:
        logger.warning(f"[compute_vib_descriptor] Invalid SMILES: {smiles}")
        return np.zeros(max_modes, dtype=float)
    # Generate 3D geometry with RDKit (classical force field)
    mol = Chem.AddHs(mol)
    try:
        AllChem.EmbedMolecule(mol, AllChem.ETKDG())
        AllChem.UFFOptimizeMolecule(mol)
    except Exception as e:
        logger.warning(f"[compute_vib_descriptor] RDKit 3D generation failed for {smiles}: {e}")
        return np.zeros(max_modes, dtype=float)
    conf = mol.GetConformer()
    n_atoms = mol.GetNumAtoms()
    coords = np.zeros((n_atoms, 3), dtype=float)
    masses = np.zeros(n_atoms, dtype=float)
    for i in range(n_atoms):
        pos = conf.GetAtomPosition(i)
        coords[i, :] = [pos.x, pos.y, pos.z]
        masses[i] = mol.GetAtomWithIdx(i).GetMass() or 1.0
    # Build harmonic Hessian from bond springs
    dim = 3 * n_atoms
    H = np.zeros((dim, dim), dtype=float)
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        ri = coords[i]
        rj = coords[j]
        diff = ri - rj
        r = np.linalg.norm(diff)
        if r < 1e-6:
            continue
        # Unit vector along the bond
        n = diff / r
        # Simple spring constant model: scaled by bond order
        order = bond.GetBondTypeAsDouble() or 1.0
        k = 1.0 * order # arbitrary units; we only need relative modes
        # 3x3 block for this bond
        Kmat = k * np.outer(n, n)
        idx_i = slice(3 * i, 3 * i + 3)
        idx_j = slice(3 * j, 3 * j + 3)
        H[idx_i, idx_i] += Kmat
        H[idx_j, idx_j] += Kmat
        H[idx_i, idx_j] -= Kmat
        H[idx_j, idx_i] -= Kmat
    # Mass-weight the Hessian: H_mw = M^{-1/2} H M^{-1/2}
    if n_atoms == 0:
        return np.zeros(max_modes, dtype=float)
    m_vec = np.repeat(masses, 3) # [m1, m1, m1, m2, m2, m2, ...]
    with np.errstate(divide="ignore"):
        inv_sqrt_m = 1.0 / np.sqrt(m_vec)
        inv_sqrt_m[~np.isfinite(inv_sqrt_m)] = 0.0
    H_mw = H * inv_sqrt_m[:, None] * inv_sqrt_m[None, :]
    # Eigen-decomposition
    try:
        eigvals, eigvecs = np.linalg.eigh(H_mw)
    except Exception as e:
        logger.warning(f"[compute_vib_descriptor] Eigen-decomposition failed for {smiles}: {e}")
        return np.zeros(max_modes, dtype=float)
    # Keep positive eigenvalues (harmonic frequencies ~ sqrt(lambda))
    positive = eigvals[eigvals > 1e-6]
    if positive.size == 0:
        logger.warning(f"[compute_vib_descriptor] No positive modes for {smiles}")
        return np.zeros(max_modes, dtype=float)
    freqs = np.sqrt(positive) # arbitrary frequency units
    freqs = np.sort(freqs)
    # Fixed-length vector
    vec = freqs[:max_modes]
    if vec.size < max_modes:
        vec = np.pad(vec, (0, max_modes - vec.size), constant_values=0.0)
    return vec.astype(float)
# ----------------------------- SIMILARITY TERMS + HYBRID SCORE -----------------------------
def similarity_terms(
    smiles: str,
    seed_fp,
    seed_vib: np.ndarray,
) -> tuple[float, float]:
    """
    Return (structural_similarity, vibrational_similarity_01) for a candidate.
    - StructuralSimilarity in [0,1]: Tanimoto on Morgan fingerprints
    - VibrationalSimilarity_01 in [0,1]:
        cosine similarity between raw harmonic descriptors of seed and candidate,
        mapped from [-1,1] -> [0,1]
    """
    mol = Chem.MolFromSmiles(smiles)
    if not mol:
        return 0.0, 0.0
    # Structural similarity (Morgan/Tanimoto)
    s_sim = FingerprintSimilarity(seed_fp, get_morgan_fp(mol)) # 0–1
    # Vibrational similarity (harmonic-mode descriptor)
    v_vec = compute_vib_descriptor(smiles)
    seed_norm = np.linalg.norm(seed_vib)
    cand_norm = np.linalg.norm(v_vec)
    if seed_norm < 1e-8 or cand_norm < 1e-8:
        v_cos = 0.0
    else:
        v_cos = float(np.dot(seed_vib, v_vec) / (seed_norm * cand_norm))
    # Map cosine similarity from [-1,1] to [0,1]
    v_sim_01 = 0.5 * (v_cos + 1.0)
    return s_sim, v_sim_01
def hybrid_score(
    smiles: str,
    seed_fp,
    seed_vib: np.ndarray,
    struct_w: float,
    vib_w: float,
    odor_w: float,
    mw_target: tuple[float, float],
    logp_min: float,
) -> float:
    """
    Hybrid score for a candidate SMILES:
    - StructuralSimilarity(S, C) in [0,1] (Tanimoto)
    - VibrationalSimilarity(S, C) in [0,1] (cosine of harmonic descriptors)
    - OdorPhyschemScore(C) in [0,1] (MW window + logP preference)
    Score = w_struct * StructuralSimilarity
          + w_vib * VibrationalSimilarity
          + w_odor * OdorPhyschemScore
    """
    s_sim, v_sim_01 = similarity_terms(smiles, seed_fp, seed_vib)
    odor_score, _, _ = odor_physchem_score(smiles, mw_target=mw_target, logp_min=logp_min)
    score = struct_w * s_sim + vib_w * v_sim_01 + odor_w * odor_score

    # Added aromatic bonus
    # Inside hybrid_score function, replace the aromatic_bonus line with:
    mol = Chem.MolFromSmiles(smiles)
    if mol:
        aromatic_rings = [
            ring for ring in Chem.GetSymmSSSR(mol)
            if all(mol.GetAtomWithIdx(idx).GetIsAromatic() for idx in ring)
        ]
        aromatic_bonus = 1.0 if len(aromatic_rings) > 0 else 0.0
    else:
        aromatic_bonus = 0.0
    score += 0.1 * aromatic_bonus  # Keep your weight

    logger.debug(
        f"[hybrid_score] SMILES={smiles} | s_sim={s_sim:.3f} | "
        f"v_sim_01={v_sim_01:.3f} | odor={odor_score:.3f} | score={score:.3f}"
    )
    return score
# ----------------------------- SAFE MUTATION GENERATOR -----------------------------
def mutate_smiles(smiles: str, n_mut=3):
    mol = Chem.MolFromSmiles(smiles)
    if not mol:
        logger.warning(f"[mutate_smiles] Invalid SMILES, cannot mutate: {smiles}")
        return []
    
    new_smiles_list = []
    for _ in range(n_mut):
        em = Chem.EditableMol(mol)
        atoms = list(range(mol.GetNumAtoms()))
        random.shuffle(atoms)
        
        mutation_type = random.choice(["replace_atom", "add_substituent", "delete_atom", "change_bond", "add_ring"])
        
        if mutation_type == "replace_atom":
            for i in atoms[:min(2, len(atoms))]:
                atom = mol.GetAtomWithIdx(i)
                if atom.GetSymbol() == "C":
                    new_sym = random.choice(["O", "N", "S", "F", "Cl"])
                    em.ReplaceAtom(i, Chem.Atom(new_sym))
        
        elif mutation_type == "add_substituent":
            if atoms:
                i = random.choice(atoms)
                atom = mol.GetAtomWithIdx(i)
                if atom.GetSymbol() == "C" and atom.GetDegree() < 4:
                    sub_smiles = random.choice(["C", "O", "[N+](=O)[O-]", "C=O"])
                    sub_mol = Chem.MolFromSmiles(sub_smiles)
                    if sub_mol:
                        new_idx = em.AddAtom(sub_mol.GetAtomWithIdx(0))
                        em.AddBond(i, new_idx, Chem.BondType.SINGLE)
        
        elif mutation_type == "delete_atom":
            terminals = [i for i in atoms if mol.GetAtomWithIdx(i).GetDegree() == 1 and not mol.GetAtomWithIdx(i).IsInRing()]
            if terminals:
                em.RemoveAtom(random.choice(terminals))
        
        elif mutation_type == "change_bond":
            bonds = list(mol.GetBonds())
            if not bonds:
                continue
            bond = random.choice(bonds)
            begin_idx = bond.GetBeginAtomIdx()
            end_idx = bond.GetEndAtomIdx()
            current_order = bond.GetBondType()
            
            # Decide new order (simple toggle; extend as needed)
            if current_order == Chem.BondType.SINGLE:
                new_order = Chem.BondType.DOUBLE
            elif current_order == Chem.BondType.DOUBLE:
                new_order = Chem.BondType.SINGLE
            elif current_order == Chem.BondType.AROMATIC:
                new_order = random.choice([Chem.BondType.SINGLE, Chem.BondType.DOUBLE])
            else:
                new_order = Chem.BondType.SINGLE  # fallback
            
            # Remove old bond, then add new one with desired order
            em.RemoveBond(begin_idx, end_idx)
            em.AddBond(begin_idx, end_idx, new_order)
            
        elif mutation_type == "add_ring":
            ring_atoms = [i for i in atoms if mol.GetAtomWithIdx(i).IsInRing() and mol.GetAtomWithIdx(i).GetIsAromatic()]
            if len(ring_atoms) >= 2:
                # Pick two adjacent aromatic atoms (improved check)
                for _ in range(10):  # Try up to 10 times to find adjacent pair
                    a1, a2 = random.sample(ring_atoms, 2)
                    if mol.GetBondBetweenAtoms(a1, a2):
                        ring_size = random.choice([5, 6])
                        last_atom = a1
                        for _ in range(ring_size - 2):
                            new_idx = em.AddAtom(Chem.Atom("C"))
                            em.AddBond(last_atom, new_idx, Chem.BondType.SINGLE)
                            last_atom = new_idx
                        em.AddBond(last_atom, a2, Chem.BondType.SINGLE)
                        break
        
        try:
            new_mol = em.GetMol()
            Chem.SanitizeMol(new_mol)
            new_smi = Chem.MolToSmiles(new_mol)
            if new_smi != smiles and new_smi not in new_smiles_list:
                new_smiles_list.append(new_smi)
                logger.debug(f"[mutate_smiles] {smiles} -> {new_smi} ({mutation_type})")
        except Exception as e:
            logger.debug(f"[mutate_smiles] Sanitization failed: {e}")
    
    return new_smiles_list  # Dedup
# ----------------------------- TABS -----------------------------
tab_gen, tab_about, tab_scoring = st.tabs(
    ["Generator", "About the Project", "Scoring Function"]
)
# ----------------------------- GENERATOR TAB -----------------------------
with tab_gen:
    st.title("SimVibe Ligand Generator")
    st.markdown("""
### **Hybrid structural + harmonic vibrational ligand generator**
SimVibe is the first publicly available tool that generates molecular analogues using a hybrid scoring function that combines **Morgan structural similarity** with **classical harmonic vibrational similarity**.
No existing structural, ML-based, or quantum-chemistry workflow integrates **normal-mode physics** directly into **generative ligand design**.
In fragrance and flavor chemistry, odor perception depends not only on molecular structure but also on **vibrational characteristics** associated with functional groups and low-frequency molecular motions. Classical harmonic normal modes provide a compact, physics-based representation of these vibrational features.
Integrating normal-mode similarity with structural similarity yields a more **biologically and perceptually relevant metric** for analogue generation than using structure alone.
SimVibe is the **first tool** to operationalize this principle in a generative ligand-design workflow.
""")
    st.image("assets/pipeline.png", width="stretch")
    with st.sidebar:
        st.markdown("### Algorithm & Scoring")
        st.caption(
            "Vibrational scoring is derived from a classical spring model on bonds "
            "and mass-weighted normal modes."
        )
        # Choose algorithm
        mode = st.radio(
            "Sampling mode",
            ["Greedy (Top-K)", "Monte Carlo"],
            index=0,
            help="Greedy: evolutionary top-k selection. Monte Carlo: Metropolis walk in hybrid space.",
        )
        # Hybrid score weights
        c1, c2, c3 = st.columns(3)
        with c1:
            struct_w = st.slider("Structure Weight", 0.0, 1.0, 0.50, 0.05)
        with c2:
            vib_w = st.slider("Vibration Weight", 0.0, 1.0, 0.30, 0.05)
        with c3:
            odor_w = st.slider(
                "Odor PhysChem Weight",
                0.0,
                1.0,
                0.20,
                0.05,
                help="Weight for MW/logP-based odorant-like profile.",
            )
        st.markdown("### Odor-Like PhysChem Target")
        mw_min, mw_max = st.slider(
            "Target MW window (Da)",
            80.0,
            300.0,
            (120.0, 200.0),
            5.0,
            help="Typical small odorants: ~120–200 Da.",
        )
        logp_min = st.slider(
            "Min logP (hydrophobicity threshold)",
            -1.0,
            5.0,
            1.0,
            0.1,
            help="Odorant ligands for ORs are usually hydrophobic (logP > 1).",
        )
        mw_target = (mw_min, mw_max)
        # Iteration controls
        n_gen = st.slider(
            "Generate / Steps per cycle", 5, 1000, 20,
            help="Greedy: new candidates per cycle. MC: steps per cycle."
        )
        n_cycles = st.slider("Cycles", 1, 1000, 20)
        # Temperature only relevant for MC
        if mode == "Monte Carlo":
            temperature = st.slider(
                "Monte Carlo Temperature",
                0.01,
                1.0,
                0.20,
                0.01,
                help="Higher T = more exploration (accept worse moves more often).",
            )
        else:
            temperature = None
    seed = st.text_input(
        "Seed SMILES (Bourgeonal – known OR1D2 agonist)",
        "CC(C)(C)c1ccc(CCC=O)cc1",
    )
    run = st.button("Run Hybrid Protocol (Harmonic Model)", type="primary")
    if run:
        seed_mol = Chem.MolFromSmiles(seed)
        if not seed_mol:
            st.error("Invalid SMILES")
            st.stop()
        logger.info(
            f"[RUN START] mode={mode} | seed={seed} | struct_w={struct_w} | "
            f"vib_w={vib_w} | odor_w={odor_w} | n_gen={n_gen} | n_cycles={n_cycles} | "
            f"temperature={temperature if mode=='Monte Carlo' else 'N/A'}"
        )
        st.info(
            "Running harmonic vibrational scoring using a classical spring model. "
        )
        # Seed descriptors (no external library)
        seed_fp = get_morgan_fp(seed_mol)
        seed_vib = compute_vib_descriptor(seed)

        # Compute seed score as reference
        seed_score = hybrid_score(
            seed, seed_fp, seed_vib, struct_w, vib_w, odor_w, mw_target, logp_min
        )
        st.info(f"Seed SMILES: {seed} | Hybrid Score: {seed_score:.3f}")
        logger.info(f"Seed score: {seed_score:.3f}")

        # Live status UI
        job_box = st.empty() # shows [x / total] & molecule counts
        status_box = st.empty() # mode + cycle/step
        detail_box = st.empty() # best/current score + SMILES
        progress = st.progress(0.0)
        if mode == "Greedy (Top-K)":
            # ----------------- GREEDY EVOLUTIONARY TOP-K MODE -----------------
            st.info("Running greedy evolutionary selection (Top-K) in hybrid space.")
            logger.info("[Greedy] Starting evolutionary Top-K run.")
            candidates = {seed}
            visited_all = {seed} # track all unique molecules ever seen
            global_best_smi = seed
            global_best_score = seed_score
            for cycle in range(n_cycles):
                # 1) Mutate some of the current candidates
                new_cands = set()
                if candidates:
                    for smi in random.sample(list(candidates), min(10, len(candidates))):
                        new_cands.update(
                            mutate_smiles(smi, n_mut=random.randint(1, 4))
                        )
                candidates.update(new_cands)
                visited_all.update(new_cands)
                # 2) Score all candidates and keep top-K
                scores = []
                for smi in candidates:
                    score = hybrid_score(
                        smi,
                        seed_fp,
                        seed_vib,
                        struct_w,
                        vib_w,
                        odor_w,
                        mw_target,
                        logp_min,
                    )
                    scores.append((smi, score))
                # Track best for this cycle
                best_smi, best_score = max(scores, key=lambda x: x[1])
                if best_score > global_best_score:
                    global_best_score = best_score
                    global_best_smi = best_smi
                # Update progress bar and text
                frac = (cycle + 1) / n_cycles
                progress.progress(frac)
                job_box.markdown(
                    f"**Job progress:** cycle **{cycle+1} / {n_cycles}** "
                    f"&nbsp;•&nbsp; unique molecules sampled: **{len(visited_all)}**"
                )
                status_box.markdown(
                    f"**Mode:** Greedy (Top-K) &nbsp;•&nbsp; "
                    f"Current candidate pool: **{len(candidates)}** molecules"
                )
                detail_box.markdown(
                    f"Best score this cycle: `{best_score:.3f}` (vs seed: `{seed_score:.3f}`) "
                    f"<br>Best SMILES: `{best_smi}`",
                    unsafe_allow_html=True,
                )
                logger.info(
                    f"[Greedy] Cycle {cycle+1}/{n_cycles} | "
                    f"pool={len(candidates)} | visited={len(visited_all)} | "
                    f"best_score={best_score:.3f} (vs seed {seed_score:.3f}) | best_smi={best_smi}"
                )
                # Keep top 60, but filter to those >= 0.7 * global_best_score (adjust threshold)
                min_threshold = 0.7 * global_best_score
                candidates = {
                    s for s, sc in sorted(scores, key=lambda x: -x[1])[:60]
                    if sc >= min_threshold
                }
                # Elitism: Always include global best
                candidates.add(global_best_smi)
            # Final scores dict for table
            visited_scores = {
                smi: hybrid_score(
                    smi,
                    seed_fp,
                    seed_vib,
                    struct_w,
                    vib_w,
                    odor_w,
                    mw_target,
                    logp_min,
                )
                for smi in candidates
            }
        else:
            # ----------------- MONTE CARLO METROPOLIS MODE -----------------
            st.info(
                "Running Metropolis Monte Carlo in hybrid space. "
                "Temperature controls how often 'worse' moves are accepted."
            )
            logger.info("[Monte Carlo] Starting Metropolis run.")
            visited_scores = {}
            # Start from seed
            current_smi = seed
            current_score = seed_score
            visited_scores[current_smi] = current_score
            global_best_smi = seed
            global_best_score = seed_score
            # Define total steps: cycles × steps per cycle (from n_gen)
            n_steps = n_cycles * n_gen
            last_delta = 0.0
            initial_T = temperature
            final_T = 0.01  # Low end for exploitation
            for step in range(n_steps):
                # Anneal temperature
                T = initial_T - (initial_T - final_T) * (step / n_steps)
                # Generate multiple proposals (e.g., 3-5) and select the one with highest score
                proposals = []
                for _ in range(3):  # Adjust for speed
                    muts = mutate_smiles(current_smi, n_mut=random.randint(1, 4))
                    if muts:
                        proposals.extend(muts)
                
                if not proposals:
                    continue
                
                # Score proposals
                prop_scores = []
                for new_smi in proposals:
                    if new_smi in visited_scores:
                        new_score = visited_scores[new_smi]
                    else:
                        new_score = hybrid_score(
                            new_smi,
                            seed_fp,
                            seed_vib,
                            struct_w,
                            vib_w,
                            odor_w,
                            mw_target,
                            logp_min,
                        )
                        visited_scores[new_smi] = new_score
                    prop_scores.append((new_smi, new_score))
                
                # Pick the best proposal (bias toward improvement)
                new_smi, new_score = max(prop_scores, key=lambda x: x[1])
                
                delta = new_score - current_score
                if delta >= 0:
                    accept = True
                else:
                    accept_prob = math.exp(delta / max(T, 1e-6))
                    accept = random.random() < accept_prob
                
                if accept:
                    current_smi = new_smi
                    current_score = new_score
                    last_delta = delta
                    if current_score > global_best_score:
                        global_best_score = current_score
                        global_best_smi = current_smi
                
                # Update progress + status
                frac = (step + 1) / n_steps
                progress.progress(frac)
                job_box.markdown(
                    f"**Job progress:** step **{step+1} / {n_steps}** "
                    f"&nbsp;•&nbsp; unique molecules visited: **{len(visited_scores)}**"
                )
                status_box.markdown(
                    f"**Mode:** Monte Carlo &nbsp;•&nbsp; "
                    f"Current SMILES: `{current_smi}`"
                )
                detail_box.markdown(
                    f"Current score: `{current_score:.3f}` (vs seed: `{seed_score:.3f}`) "
                    f"&nbsp;•&nbsp; Last Δ: `{last_delta:.3f}`",
                    unsafe_allow_html=True,
                )
                # Log periodically to avoid spamming every step
                if step == 0 or step == n_steps - 1 or (step + 1) % 10 == 0:
                    logger.info(
                        f"[Monte Carlo] Step {step+1}/{n_steps} | "
                        f"visited={len(visited_scores)} | "
                        f"current_score={current_score:.3f} (vs seed {seed_score:.3f}) | delta={last_delta:.3f} | "
                        f"current_smi={current_smi}"
                    )
            st.success(
                f"Monte Carlo finished: visited {len(visited_scores)} unique candidates"
            )
        progress.empty()
        # ----------------- BUILD RESULT TABLE (COMMON TO BOTH MODES) -----------------
        # Take top 30 by hybrid score
        top_items = sorted(
            visited_scores.items(),
            key=lambda x: -x[1]
        )[:30]
        top_smiles = [smi for smi, _ in top_items]
        struct_sims = []
        vib_sims = []
        hybrid_scores = []
        odor_scores = []
        mws = []
        logps = []
        for smi in top_smiles:
            s_sim, v_sim_01 = similarity_terms(smi, seed_fp, seed_vib)
            odor_score, mw, logp = odor_physchem_score(
                smi, mw_target=mw_target, logp_min=logp_min
            )
            struct_sims.append(s_sim)
            vib_sims.append(v_sim_01)
            hybrid_scores.append(visited_scores[smi])
            odor_scores.append(odor_score)
            mws.append(mw)
            logps.append(logp)
        result = pd.DataFrame({
            "SMILES": top_smiles,
            "Hybrid Score": hybrid_scores,
            "Structural Sim": struct_sims,
            "Vibrational Sim": vib_sims,
            "Odor PhysChem": odor_scores,
            "MW": mws,
            "logP": logps,
        })
        result = result.sort_values("Hybrid Score", ascending=False)
        logger.info(
            f"[RUN END] mode={mode} | top_n={len(result)} | "
            f"best_score={result['Hybrid Score'].iloc[0]:.3f} | "
            f"best_smi={result['SMILES'].iloc[0]}"
        )
        st.dataframe(result, width="stretch")
        # Molecule depictions
        imgs = []
        for _, row in result.head(12).iterrows():
            mol = Chem.MolFromSmiles(row["SMILES"])
            if mol:
                legend = (
                    f"H={row['Hybrid Score']:.3f} | "
                    f"S={row['Structural Sim']:.3f} | "
                    f"V={row['Vibrational Sim']:.3f} | "
                    f"O={row['Odor PhysChem']:.3f}"
                )
                imgs.append(Draw.MolToImage(mol, size=(300, 300), legend=legend))
        if imgs:
            st.image(imgs, width=320)
        # Download CSV
        filename = (
            "SimVibe_MC_analogues.csv"
            if mode == "Monte Carlo"
            else "SimVibe_Greedy_analogues.csv"
        )
        csv = result.to_csv(index=False)
        st.download_button(
            "Download Results",
            csv,
            filename,
            "text/csv"
        )
st.markdown("""
## **References**
1. Hollas, J. M. *Modern Spectroscopy*, 4th ed. Wiley (2004).
2. Wilson, E. B.; Decius, J. C.; Cross, P. C. *Molecular Vibrations.* Dover (1980).
3. Block, E. et al. “Implausibility of the Vibrational Theory of Olfaction.” *PNAS* (2015).
4. Gane, S. et al. “Molecular Vibration-Sensing Component in Human Olfaction.” *PNAS* (2013).
5. Zarzo, M.; Stanton, D. T. “Functional Group Contributions to Fragrance Perception.”
   *J. Agric. Food Chem.* (2006).
6. Rogers, D.; Hahn, M. “Extended-Connectivity Fingerprints.” *J. Chem. Inf. Model.* (2010).
7. Jensen, J. H. “A Graph-Based Genetic Algorithm for Molecular Discovery.” *arXiv:1904.01268* (2019).
8. Keller, A.; Vosshall, L. B. “Olfaction: Odor Object Perception Depends on More Than Molecular Structure.”
   *Science* (2017).
""")
# ----------------------------- ABOUT THE PROJECT TAB -----------------------------
with tab_about:
    st.title("About the Project")
    st.markdown(
        '''
## Hybrid Structure–Vibration Similarity with Classical Harmonic Modes
This project introduces a general-purpose framework for molecular analogue generation that
integrates **structural similarity** with **vibrational information** derived from a
classical harmonic approximation.
Instead of performing quantum-chemical calculations (DFT, HF, etc.), we:
- Embed and optimize 3D geometries with a **molecular mechanics force field** (RDKit/UFF)
- Treat each **bond as a harmonic spring** with a simple, bond-order–scaled force constant
- Construct a **mass-weighted Hessian** for the bonded network
- Perform **normal mode analysis** to obtain harmonic frequencies (up to a fixed number of modes)
These modes capture how atoms move collectively around the equilibrium geometry under a
spring-like approximation. The resulting frequency spectrum is used as a compact
vibrational descriptor.
We combine this with standard structural fingerprints to define a **hybrid similarity
metric** aimed explicitly at finding **analogues that are simultaneously close in structure
and in vibrational behavior**.
### Applications
This framework can support diverse research areas, including:
- Odorant and flavor chemistry
- GPCR ligand exploration in low-data domains
- SAR and analogue series generation
- Exploratory medicinal chemistry
- Spectroscopy-inspired chemoinformatics
- Automated hypothesis generation for chemical biology
### Methodological Summary
| Component | Implementation | Comment |
|----------|----------------|---------|
| Structural descriptor | Morgan/ECFP fingerprints | Standard chemoinformatics similarity |
| Vibrational descriptor | Harmonic spring model + normal modes | No quantum chemistry; bonded network only |
| Geometry | RDKit ETKDG + UFF | Fast, robust for small organics |
| Chemical validation | RDKit sanitization | Ensures chemically valid structures |
| Optimizer | Greedy Top-K + Metropolis Monte Carlo | Exploitation + stochastic exploration |
### Significance
This project provides a transparent and extensible framework for **multi-modal molecular
similarity** that remains computationally light: all vibrational features are derived from
a classical spring model, without any ab initio electronic-structure calculations. It is
explicitly designed to explore chemical space for **candidates that preserve both structural
and vibrational similarity to a chosen seed ligand**.
'''
    )
# ----------------------------- SCORING FUNCTION TAB -----------------------------
with tab_scoring:
    st.title("Scoring Function")
    st.markdown(
        r'''
## Hybrid Scoring Framework: Structural + Harmonic Vibrational + Odor PhysChem Similarity
The goal of this tool is to identify **analogues that are simultaneously similar in
structure, in vibrational behavior, and in odorant-like physchem profile** to a seed molecule.
Each generated molecule is evaluated using a **hybrid similarity score** that integrates:
1. **Structural similarity**, based on extended-connectivity fingerprints
2. **Vibrational similarity**, derived from a classical harmonic spring model and
   normal mode analysis
3. **Odor physchem similarity**, based on molecular weight and hydrophobicity (logP)
   consistent with small, hydrophobic odorant ligands
All similarity terms are normalized to the range **0–1** before combination.
---
## 1. Structural Similarity Component
Structural similarity is computed using **Morgan/ECFP fingerprints** (radius = 2, 2048 bits),
a widely used circular fingerprint method that encodes atom neighborhoods and functional
group environments.
For the seed molecule (*S*) and a candidate molecule (*C*), we compute:
\[
\text{StructuralSimilarity}(S, C) = \text{Tanimoto}(FP_S, FP_C)
\]
where
\[
\text{Tanimoto}(A, B) = \frac{|A \cap B|}{|A| + |B| - |A \cap B|}
\]
This yields a similarity value in the range **[0,1]**, with 1 indicating identical
fingerprints.
---
## 2. Harmonic Vibrational Similarity Component
For each molecule, we:
1. Generate a 3D geometry from SMILES using RDKit (ETKDG + UFF).
2. Treat every covalent bond as a **harmonic spring** with a bond-order–scaled force constant.
3. Build the **mass-weighted Hessian** of the bonded network (3N × 3N).
4. Diagonalize the Hessian to obtain normal mode eigenvalues.
5. Convert positive eigenvalues to “frequencies” via \(\omega_i \propto \sqrt{\lambda_i}\), sort,
   and assemble a fixed-length descriptor vector.
We compare the seed (*S*) and candidate (*C*) harmonic vibrational descriptors
\(\vec{v}_S\) and \(\vec{v}_C\) using cosine similarity:
\[
\cos\theta
= \frac{\vec{v}_S \cdot \vec{v}_C}
       {\left\lVert \vec{v}_S \right\rVert
        \left\lVert \vec{v}_C \right\rVert}
\]
This is mapped from \([-1, 1]\) to \([0, 1]\):
\[
\text{VibrationalSimilarity}(S, C) = \frac{\cos\theta + 1}{2}.
\]
---
## 3. Odor PhysChem Similarity Component
To bias toward **odorant-like ligands**, we use a simple physchem model based on:
- Molecular weight (**MW**) within a tunable window \([MW_{\min}, MW_{\max}]\), with a
  peak score at the center and 0 at the edges (triangular membership).
- Hydrophobicity (**logP**) above a user-defined threshold, ramping up and
  saturating at moderate logP.
The resulting **OdorPhysChem score** lies in \([0,1]\), where 1 indicates a molecule that
simultaneously sits near the center of the MW window and exceeds the logP threshold.
---
## 4. Final Hybrid Score
The final hybrid score used for selection (Greedy Top-K) or sampling (Monte Carlo) is a
**linear combination** of the three normalized similarity terms:
\[
\text{Score}(S, C)
= w_{\text{struct}}
  \cdot \text{StructuralSimilarity}(S, C)
+ w_{\text{vib}}
  \cdot \text{VibrationalSimilarity}(S, C)
+ w_{\text{odor}}
  \cdot \text{OdorPhysChem}(C)
\]
where \(w_{\text{struct}}, w_{\text{vib}}, w_{\text{odor}} \ge 0\) are user-chosen weights.
Because all components are in **[0,1]**, the weights directly express the relative
importance of **structural**, **vibrational**, and **odorant-like physchem** similarity in
the search for analogues. High-scoring candidates are, by construction, those that
simultaneously align with the seed in:
- Morgan fingerprint space
- Harmonic vibrational descriptor space
- MW/logP-defined odorant physchem space
'''
    )
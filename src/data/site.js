export const siteMeta = {
  name: "Shaoyang Guo",
  chineseName: "郭绍阳",
  title:
    "Physics undergraduate at PKU, working on VLM post-training and STEM reasoning evaluation.",
  summary:
    "I am a physics undergraduate at Peking University (Class of 2027). My research sits at the intersection of vision-language model post-training, benchmark design, and data-centric optimization for scientific reasoning.",
  email: "guoshaoyang@stu.pku.edu.cn",
  github: "https://github.com/guoshaoyang-pku",
  location: "Beijing, China",
};

export const quickFacts = [
  {
    label: "Current",
    value: "Research Intern at ByteDance Seed, VLM Post-Training",
  },
  {
    label: "Education",
    value: "PKU School of Physics, Excellence Program (CPhO Gold)",
  },
  {
    label: "Publications",
    value: "PHYBench (arXiv:2504.16074) & VLA Survey (arXiv:2507.01925)",
  },
  {
    label: "Interests",
    value: "Physics of AI, Embodied Intelligence, VLM",
  },
];

export const aboutParagraphs = [
  "I entered Peking University through the Excellence Program after winning a Gold Medal at the Chinese Physics Olympiad. My academic background spans quantum mechanics, quantum field theory, statistical physics, and computational methods.",
  "Since 2025 I have been working as a research intern at ByteDance Seed, contributing to multi-stage post-training pipelines (RL, SFT, mid-training) for vision-language models, with emphasis on improving STEM reasoning capabilities.",
  "I care about rigorous evaluation and data quality — my work on PHYBench and large-scale data curation reflects this focus on making AI systems more reliable in scientific domains.",
];

export const blogs = [
  {
    title: "What makes a STEM benchmark actually useful?",
    eyebrow: "Planned Essay",
    image: "/images/project-phybench.svg",
    summary:
      "Notes on building evaluations that reveal real reasoning capability rather than benchmark-specific pattern matching, with lessons from PHYBench.",
    tags: ["Benchmarking", "Physics", "Evaluation"],
    link: null,
  },
  {
    title: "Views on large model training",
    eyebrow: "Writing Plan",
    image: "/images/project-vlm-post-training.svg",
    summary:
      "A continuing series for organizing personal views on post-training, data quality, RL/SFT dynamics, and the practical craft of making models better.",
    tags: ["Post-Training", "Data Quality", "VLM"],
    link: null,
  },
  {
    title: "From physics olympiad to AI research",
    eyebrow: "Personal Note",
    image: "/images/project-vla-survey.svg",
    summary:
      "Reflections on how physics training shapes taste in AI research: problem selection, abstraction, experiments, and long-term curiosity.",
    tags: ["Research", "Physics", "Personal"],
    link: null,
  },
];

export const researchArtifacts = [
  {
    title: "PHYBench",
    type: "Paper (NeurIPS 2025 submission)",
    href: "https://arxiv.org/abs/2504.16074",
    description:
      "Holistic Evaluation of Physical Perception and Reasoning in Large Language Models. 500 original physics problems, 178 contributors.",
  },
  {
    title: "VLA Survey",
    type: "Survey Paper",
    href: "https://arxiv.org/abs/2507.01925",
    description:
      "A Survey on Vision-Language-Action Models: An Action Tokenization Perspective.",
  },
  {
    title: "Awesome-VLA-Papers",
    type: "GitHub Repository",
    href: "https://github.com/Psi-Robot/Awesome-VLA-Papers",
    description:
      "Curated paper list for VLA models, organized by action representation methodology.",
  },
];

export const experience = [
  {
    role: "Research Intern, Seed VLM Post-Training Team",
    org: "ByteDance",
    period: "Jul 2025 – Present",
    description:
      "Working on multi-stage post-training for vision-language models, targeting STEM reasoning capabilities through RL, SFT, and mid-training.",
    bullets: [
      "Responsible for multi-stage delivery across RL, SFT, and mid-training; participated in early mid-training exploration.",
      "Researched BoN sampling and sample-repeat strategies; disproved equilibrium-based sampling assumptions; established RL–SFT equivalence under repeat conditions.",
      "Developed large-scale data cleaning pipeline with probabilistic quality models; built textbook exercise extraction system (100M+ QA pairs).",
      "Exploring TransferRL (combining SFT with RL) and CoT compression methods for efficient reasoning synthesis.",
    ],
  },
  {
    role: "Co-initiator & Co-first Author, PHYBench",
    org: "Peking University (Eureka Lab)",
    period: "Feb 2025 – Sep 2025",
    description:
      "Co-initiated and co-led PHYBench, a physics reasoning benchmark for LLMs. Paper submitted to NeurIPS 2025.",
    bullets: [
      "Identified gaps in existing LLM physics evaluation; led project from concept validation to full data pipeline.",
      "Organized 178 PKU students to build 500 high-quality original physics problems in 2 weeks.",
      "Designed evaluation criteria, quality control processes, and failure mode analysis frameworks.",
      "Completed main experiments and analysis of frontier LLM performance across physics subdomains.",
    ],
  },
  {
    role: "Research Assistant, VLA Survey",
    org: "PsiRobot Lab, Peking University",
    period: "Mar 2025 – Aug 2025",
    description:
      "Co-authored a survey on Vision-Language-Action models. Advisor: Prof. Yaodong Yang.",
    bullets: [
      "Responsible for the Raw Action chapter; reviewed 30+ key papers on end-to-end VLA architectures.",
      "Organized taxonomies for VLA model design from an action tokenization perspective.",
    ],
  },
];

export const educationAndHonors = [
  {
    title: "Peking University, School of Physics",
    detail:
      "B.S. in Physics (expected 2027). Admitted via Excellence Program (CPhO Gold Medal). Top 10% in first-year physics cohort. Completed 141/149 credits by sophomore year including 3 graduate courses.",
  },
  {
    title: "National Scholarship (2024)",
    detail:
      "Ministry of Education, top 1% at Peking University.",
  },
  {
    title: "Chinese Physics Olympiad Gold Medal",
    detail:
      "National rank #57 (2022). Admitted to PKU Physics via Excellence Program.",
  },
  {
    title: "NOIP First Prize (2020)",
    detail:
      "National Olympiad in Informatics in Provinces.",
  },
];

export const writingIdeas = [
  {
    title: "What makes a STEM benchmark actually useful?",
    status: "Planned note",
  },
  {
    title: "BoN sampling dynamics: what we learned",
    status: "Planned note",
  },
  {
    title: "From physics olympiad to AI research",
    status: "Planned note",
  },
];

export const contactLinks = [
  {
    label: "GitHub",
    href: "https://github.com/guoshaoyang-pku",
    value: "github.com/guoshaoyang-pku",
  },
  {
    label: "Email",
    href: "mailto:guoshaoyang@stu.pku.edu.cn",
    value: "guoshaoyang@stu.pku.edu.cn",
  },
  {
    label: "CV Summary",
    href: "/cv/",
    value: "View the public-facing CV page",
  },
];

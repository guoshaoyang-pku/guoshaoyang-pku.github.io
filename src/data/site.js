export const siteMeta = {
  name: "Shaoyang Guo",
  chineseName: "郭绍阳",
  title:
    "Physics undergraduate at PKU, working on LLM/VLM post-training, agents, benchmarking, and Physics of AI.",
  summary:
    "I am a physics undergraduate at Peking University (Class of 2027). My work focuses on LLM/VLM post-training, RL/SFT, agentic research systems, benchmarking, and Physics of AI.",
  email: "guoshaoyang@stu.pku.edu.cn",
  github: "https://github.com/guoshaoyang-pku",
  cvPdf: "/cv/guoshaoyang-pku-cv-v5.pdf",
  location: "Beijing, China",
};

export const quickFacts = [
  {
    label: "Current",
    value: "Research Intern at ByteDance Seed, VLM/LLM Post-Training",
  },
  {
    label: "Education",
    value: "PKU School of Physics, B.S. expected 2027",
  },
  {
    label: "Publications",
    value: "PHYBench (arXiv:2504.16074) & VLA Survey (arXiv:2507.01925)",
  },
  {
    label: "Interests",
    value: "LLM/VLM, RL/SFT/Agents, Benchmarking, Physics of AI",
  },
];

export const aboutParagraphs = [
  "I am a physics undergraduate at Peking University, admitted through the PKU Excellence Program after winning a Chinese Physics Olympiad Gold Medal. My academic record includes a 3.74/4.00 GPA, top 10% standing in the School of Physics, and 141/149 credits completed by sophomore year including 3 graduate courses.",
  "Since July 2025, I have been a research intern at ByteDance Seed, working on VLM/LLM post-training across RL, SFT, mid-training, rollout pipelines, data systems, and agentic research workflows.",
  "My research interests center on LLM/VLM, RL/SFT/Agents, benchmarking, and Physics of AI: building stronger models while understanding how to evaluate and improve scientific reasoning.",
];

export const blogs = [
  {
    title: "ArchitectureIQ 项目全面 Review",
    eyebrow: "Project Review",
    image: "/images/project-vlm-post-training.svg",
    summary:
      "A comprehensive review of ArchitectureIQ's question-generation pipeline, significance tests, evaluation protocol, meta-model results, conclusions, and open problems.",
    tags: ["ArchitectureIQ", "Evaluation", "Meta-Model"],
    link: "/blogs/architectureiq-project-review/",
  },
  {
    title: "N-gram Gap 机制指南",
    eyebrow: "Mechanism Guide",
    image: "/blogs/ngram-gap-mechanism-guide/fig14_dist_ctx.svg",
    summary:
      "A visual guide to the N-gram Gap mechanism, including global N-gram frequency, validation loss, contribution analysis, and the training cliff.",
    tags: ["N-gram", "Language Models", "Mechanistic Analysis"],
    link: "/blogs/ngram-gap-mechanism-guide/",
  },
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
    type: "arXiv:2504.16074, NeurIPS 2025 submission",
    href: "https://arxiv.org/abs/2504.16074",
    description:
      "Holistic Evaluation of Physical Perception and Reasoning in Large Language Models. Co-initiated the project and helped organize 178 PKU students to create 500 original physics problems.",
  },
  {
    title: "VLA Survey",
    type: "arXiv:2507.01925",
    href: "https://arxiv.org/abs/2507.01925",
    description:
      "A Survey on Vision-Language-Action Models: An Action Tokenization Perspective. Responsible for the Raw Action chapter and 30+ related papers.",
  },
];

export const experience = [
  {
    role: "Research Intern, VLM/LLM Post-Training",
    org: "ByteDance Seed",
    period: "Jul 2025 – Present",
    description:
      "Working on post-training and research automation for VLM/LLM systems, with emphasis on RL, SFT, mid-training, rollouts, data pipelines, and agent workflows.",
    bullets: [
      "Contributed to HiPhO-oriented RL, SFT, and mid-training work for Seed 2.0 models, improving reported Lite performance from 72.5 to 83.8.",
      "Participated in mid-training runs at large compute scale and supported rollout pipelines for model improvement.",
      "Built data and prompt pipelines for QA pairs, CoT compression, summaries, and SFT-to-RL transfer experiments.",
      "Explored auto-research agent loops, adversarial pair agents, and agent-based research settings.",
    ],
  },
  {
    role: "Co-initiator & Co-first Author, PHYBench",
    org: "Peking University, Eureka Lab",
    period: "Feb 2025 – Sep 2025",
    description:
      "Co-initiated and co-led PHYBench, a physics perception and reasoning benchmark for LLMs.",
    bullets: [
      "Identified gaps in existing LLM physics evaluation and led the project from concept validation to a full data pipeline.",
      "Organized 178 PKU students to build 500 high-quality original physics problems in 2 weeks.",
      "Designed evaluation criteria and quality-control workflows for LLM physics reasoning.",
      "Co-authored the arXiv preprint submitted to NeurIPS 2025.",
    ],
  },
  {
    role: "Research Assistant, VLA Survey",
    org: "PsiRobot Lab, Peking University",
    period: "Mar 2025 – Aug 2025",
    description:
      "Co-authored a survey on Vision-Language-Action models from an action-tokenization perspective.",
    bullets: [
      "Responsible for the Raw Action chapter; reviewed 30+ key papers on end-to-end VLA architectures.",
      "Organized taxonomies for VLA model design and contributed to the arXiv preprint.",
    ],
  },
];

export const educationAndHonors = [
  {
    title: "Peking University, School of Physics",
    detail:
      "B.S. in Physics, expected Jun 2027. GPA 3.74/4.00, top 10% in the School of Physics; completed 141/149 credits by sophomore year including 3 graduate courses.",
  },
  {
    title: "National Scholarship (2024)",
    detail:
      "Ministry of Education, top 1% at Peking University.",
  },
  {
    title: "Chinese Physics Olympiad Gold Medal",
    detail:
      "National rank #57 (2022). Admitted to PKU Physics via PKU Excellence Program.",
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
    label: "Download CV",
    href: siteMeta.cvPdf,
    value: "guoshaoyang-pku-cv-v5.pdf",
    download: true,
  },
];

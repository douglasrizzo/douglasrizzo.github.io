// get the ninja-keys element
const ninja = document.querySelector('ninja-keys');

// add the home and posts menu items
ninja.data = [{
    id: "nav-about",
    title: "about",
    section: "Navigation",
    handler: () => {
      window.location.href = "/";
    },
  },{id: "nav-blog",
          title: "blog",
          description: "",
          section: "Navigation",
          handler: () => {
            window.location.href = "/blog/";
          },
        },{id: "nav-publications",
          title: "publications",
          description: "publications by categories in reversed chronological order.",
          section: "Navigation",
          handler: () => {
            window.location.href = "/publications/";
          },
        },{id: "nav-projects",
          title: "projects",
          description: "A growing collection of my cool projects.",
          section: "Navigation",
          handler: () => {
            window.location.href = "/projects/";
          },
        },{id: "nav-repositories",
          title: "repositories",
          description: "",
          section: "Navigation",
          handler: () => {
            window.location.href = "/repositories/";
          },
        },{id: "nav-cv",
          title: "cv",
          description: "",
          section: "Navigation",
          handler: () => {
            window.location.href = "/cv/";
          },
        },{id: "post-building-movie-recommendation-systems-using-the-movielens-tmdb-datasets",
      
        title: "Building movie recommendation systems using the MovieLens + TMDB datasets",
      
      description: "A showcase of TF-IDF, bi-encoders and cross-encoders",
      section: "Posts",
      handler: () => {
        
          window.location.href = "/blog/2025/03/text-based-movie-recs/";
        
      },
    },{id: "post-running-a-gemma-powered-question-answering-chatbot-locally-with-langchain-ollama",
      
        title: "Running a Gemma-powered question-answering chatbot locally with LangChain + Ollama",
      
      description: "",
      section: "Posts",
      handler: () => {
        
          window.location.href = "/blog/2024/03/local-gemma-chatbot-langchain-ollama/";
        
      },
    },{id: "post-answering-questions-from-an-obsidian-database-with-llms-rag",
      
        title: "Answering questions from an Obsidian database with LLMs + RAG",
      
      description: "",
      section: "Posts",
      handler: () => {
        
          window.location.href = "/blog/2024/02/llm-qa-obsidian-rag/";
        
      },
    },{id: "post-zotero-tips-and-tricks",
      
        title: "Zotero tips and tricks",
      
      description: "",
      section: "Posts",
      handler: () => {
        
          window.location.href = "/blog/2021/06/zotero-tips/";
        
      },
    },{id: "post-resources-to-self-study-mathematics-for-machine-learning",
      
        title: "Resources to self-study mathematics for machine learning",
      
      description: "",
      section: "Posts",
      handler: () => {
        
          window.location.href = "/blog/2021/06/mathematics-self-study/";
        
      },
    },{id: "post-configuring-visual-studio-code-for-latex",
      
        title: "Configuring Visual Studio Code for LaTeX",
      
      description: "",
      section: "Posts",
      handler: () => {
        
          window.location.href = "/blog/2021/06/configuring-vscode-for-latex/";
        
      },
    },{id: "post-detecting-soccer-balls-with-reduced-neural-networks",
      
        title: "Detecting soccer balls with reduced neural networks",
      
      description: "",
      section: "Posts",
      handler: () => {
        
          window.location.href = "/blog/2021/02/detecting-soccer-balls-with-reduced-neural-networks/";
        
      },
    },{id: "post-razões-para-deep-reinforcement-learning-não-funcionar",
      
        title: "Razões para deep reinforcement learning não funcionar",
      
      description: "",
      section: "Posts",
      handler: () => {
        
          window.location.href = "/blog/2021/02/razoes-para-deep-reinforcement-learning-nao-funcionar/";
        
      },
    },{id: "post-classificação-da-base-de-dados-iris-redes-menores-e-regularização",
      
        title: "Classificação da base de dados Iris - redes menores e regularização",
      
      description: "",
      section: "Posts",
      handler: () => {
        
          window.location.href = "/blog/2020/08/regularizacao-microrede/";
        
      },
    },{id: "post-classificação-da-base-de-dados-iris-utilizando-redes-neurais-e-pca",
      
        title: "Classificação da base de dados Iris utilizando redes neurais e PCA",
      
      description: "",
      section: "Posts",
      handler: () => {
        
          window.location.href = "/blog/2020/08/iris-pca-keras/";
        
      },
    },{id: "post-classificação-da-base-de-dados-iris-utilizando-um-perceptron-multi-camadas-em-keras",
      
        title: "Classificação da base de dados Iris utilizando um perceptron multi-camadas em Keras",
      
      description: "",
      section: "Posts",
      handler: () => {
        
          window.location.href = "/blog/2020/08/iris-keras/";
        
      },
    },{id: "post-how-to-train-your-own-object-detection-models-using-the-tensorflow-object-detection-api-2020-update",
      
        title: "How to train your own object detection models using the TensorFlow Object Detection...",
      
      description: "",
      section: "Posts",
      handler: () => {
        
          window.location.href = "/blog/2020/08/tf-obj-tutorial/";
        
      },
    },{id: "post-open-source-contributions-during-my-phd",
      
        title: "Open source contributions during my PhD",
      
      description: "",
      section: "Posts",
      handler: () => {
        
          window.location.href = "/blog/2020/08/phd-oss-contributions/";
        
      },
    },{id: "post-reverse-engineering-a-step-decay-for-learning-rate",
      
        title: "Reverse engineering a step decay for learning rate",
      
      description: "",
      section: "Posts",
      handler: () => {
        
          window.location.href = "/blog/2020/07/step-decay-lr/";
        
      },
    },{id: "post-using-task-spooler-to-queue-experiments-on-linux",
      
        title: "Using task-spooler to queue experiments on Linux",
      
      description: "",
      section: "Posts",
      handler: () => {
        
          window.location.href = "/blog/2020/07/ts-queue-experiments/";
        
      },
    },{id: "post-como-acessar-a-vpn-da-fei-no-linux-usando-o-openfortivpn",
      
        title: "Como acessar a VPN da FEI no Linux usando o openfortivpn",
      
      description: "",
      section: "Posts",
      handler: () => {
        
          window.location.href = "/blog/2020/07/fei-openfortivpn/";
        
      },
    },{id: "post-mapping-numpad-keys-with-sxhkd",
      
        title: "Mapping Numpad keys with sxhkd",
      
      description: "",
      section: "Posts",
      handler: () => {
        
          window.location.href = "/blog/2020/05/sxkhd-masterkeys-pro-m-abnt2/";
        
      },
    },{id: "post-in-c-classes-and-structs-are-the-same-thing",
      
        title: "In C++, classes and structs are the same thing",
      
      description: "",
      section: "Posts",
      handler: () => {
        
          window.location.href = "/blog/2020/05/cpp-structs-classes/";
        
      },
    },{id: "post-solving-the-mistery-of-the-kl-divergence",
      
        title: "Solving the mistery of the KL divergence",
      
      description: "",
      section: "Posts",
      handler: () => {
        
          window.location.href = "/blog/2020/05/kl-div-pytorch/";
        
      },
    },{id: "post-approximating-pi-using-euler-39-s-identity-with-primes",
      
        title: "Approximating $$\pi$$ using Euler&#39;s identity with primes",
      
      description: "",
      section: "Posts",
      handler: () => {
        
          window.location.href = "/blog/2020/05/euler-pi/";
        
      },
    },{id: "post-sieve-of-eratosthenes",
      
        title: "Sieve of Eratosthenes",
      
      description: "",
      section: "Posts",
      handler: () => {
        
          window.location.href = "/blog/2020/05/eratosthenes/";
        
      },
    },{id: "post-visualizing-temperature-in-a-boltzmann-policy",
      
        title: "Visualizing temperature in a Boltzmann policy",
      
      description: "",
      section: "Posts",
      handler: () => {
        
          window.location.href = "/blog/2020/05/boltzmann-policy-temperature/";
        
      },
    },{id: "post-free-online-resources-to-study-reinforcement-learning-and-deep-rl",
      
        title: "Free online resources to study reinforcement learning and deep RL",
      
      description: "",
      section: "Posts",
      handler: () => {
        
          window.location.href = "/blog/2020/05/rl-resources/";
        
      },
    },{id: "post-installing-cuda-10-1-and-cudnn-7-6-on-manjaro-linux",
      
        title: "Installing CUDA 10.1 and cuDNN 7.6 on Manjaro Linux",
      
      description: "",
      section: "Posts",
      handler: () => {
        
          window.location.href = "/blog/2020/03/cuda-manjaro/";
        
      },
    },{id: "post-graph-neural-network-libraries",
      
        title: "Graph neural network libraries",
      
      description: "",
      section: "Posts",
      handler: () => {
        
          window.location.href = "/blog/2020/02/gnn-libraries/";
        
      },
    },{id: "post-dunn-index-for-clusters-analysis",
      
        title: "Dunn index for clusters analysis",
      
      description: "",
      section: "Posts",
      handler: () => {
        
          window.location.href = "/blog/2019/08/dunn-index/";
        
      },
    },{id: "post-dodo-detector-single-shot-detector-example",
      
        title: "dodo_detector Single Shot Detector example",
      
      description: "",
      section: "Posts",
      handler: () => {
        
          window.location.href = "/blog/2019/03/dodo-detector-ssd/";
        
      },
    },{id: "post-what-39-s-the-longest-word-you-can-write-with-seven-segment-displays",
      
        title: "What&#39;s The Longest Word You Can Write With Seven-Segment Displays?",
      
      description: "",
      section: "Posts",
      handler: () => {
        
          window.location.href = "/blog/2018/10/seven-segment-display/";
        
      },
    },{id: "books-the-godfather",
          title: 'The Godfather',
          description: "",
          section: "Books",handler: () => {
              window.location.href = "/books/the_godfather/";
            },},{id: "projects-catsim",
          title: 'catsim',
          description: "computerized adaptive testing simulation in Python",
          section: "Projects",handler: () => {
              window.location.href = "/projects/catsim/";
            },},{id: "projects-c-algorithms",
          title: 'c++ algorithms',
          description: "ML, linear algebra and numerical analysis algorithms in C++",
          section: "Projects",handler: () => {
              window.location.href = "/projects/cpp_algos/";
            },},{id: "projects-dodo-detector",
          title: 'dodo detector',
          description: "Object detection for robotics with Python and ROS",
          section: "Projects",handler: () => {
              window.location.href = "/projects/dodo_detector/";
            },},{id: "projects-fei-latex-class",
          title: 'FEI LaTeX class',
          description: "LaTeX class for dissertations and theses",
          section: "Projects",handler: () => {
              window.location.href = "/projects/fei-latex-class/";
            },},{id: "projects-fruit-detection-showcase",
          title: 'fruit detection showcase',
          description: "training convolutional neural networks do detect oranges",
          section: "Projects",handler: () => {
              window.location.href = "/projects/fruit_detection_showcase/";
            },},{id: "projects-jcat",
          title: 'jCAT',
          description: "Java Web application for computerized adaptive testing application in tablets",
          section: "Projects",handler: () => {
              window.location.href = "/projects/jcat/";
            },},{id: "projects-generating-faces-using-gans",
          title: 'generating faces using gans',
          description: "training convolutional neural networks do generate faces",
          section: "Projects",handler: () => {
              window.location.href = "/projects/wgan_celeba/";
            },},{
        id: 'social-github',
        title: 'GitHub',
        section: 'Socials',
        handler: () => {
          window.open("https://github.com/douglasrizzo", "_blank");
        },
      },{
        id: 'social-lattes',
        title: 'Lattes',
        section: 'Socials',
        handler: () => {
          window.open("http://lattes.cnpq.br/8493360634720203", "_blank");
        },
      },{
        id: 'social-linkedin',
        title: 'LinkedIn',
        section: 'Socials',
        handler: () => {
          window.open("https://www.linkedin.com/in/douglasrizzo", "_blank");
        },
      },{
        id: 'social-orcid',
        title: 'ORCID',
        section: 'Socials',
        handler: () => {
          window.open("https://orcid.org/0000-0002-0478-467X", "_blank");
        },
      },{
        id: 'social-researchgate',
        title: 'ResearchGate',
        section: 'Socials',
        handler: () => {
          window.open("https://www.researchgate.net/profile/Douglas-De-Rizzo-Meneghetti/", "_blank");
        },
      },{
        id: 'social-rss',
        title: 'RSS Feed',
        section: 'Socials',
        handler: () => {
          window.open("/feed.xml", "_blank");
        },
      },{
        id: 'social-scholar',
        title: 'Google Scholar',
        section: 'Socials',
        handler: () => {
          window.open("https://scholar.google.com/citations?user=V30JReAAAAAJ", "_blank");
        },
      },{
        id: 'social-stackoverflow',
        title: 'Stackoverflow',
        section: 'Socials',
        handler: () => {
          window.open("https://stackoverflow.com/users/1245214", "_blank");
        },
      },{
      id: 'light-theme',
      title: 'Change theme to light',
      description: 'Change the theme of the site to Light',
      section: 'Theme',
      handler: () => {
        setThemeSetting("light");
      },
    },
    {
      id: 'dark-theme',
      title: 'Change theme to dark',
      description: 'Change the theme of the site to Dark',
      section: 'Theme',
      handler: () => {
        setThemeSetting("dark");
      },
    },
    {
      id: 'system-theme',
      title: 'Use system default theme',
      description: 'Change the theme of the site to System Default',
      section: 'Theme',
      handler: () => {
        setThemeSetting("system");
      },
    },];

// Site-wide metadata: group info, page navigation, and external links.
// This file does NOT contain task-level or assignment-level report content.
// Report content is loaded dynamically from each section's README.md.
const CONTENT = {
  site: {
    group: "Group13",
    supervisor: "Dr. Lê Thành Sách",
    members: [
      "Lê Đức Phương - 2570480",
      "Nguyễn Đình Khánh - 2570227",
      "Nguyễn Huy Hoàng - 2570089",
      "Nguyễn Huỳnh Như - 2570471"
    ]
  },
  pages: [
    {
      id: "assignment_1",
      title: "Assignment 1",
      subtitle: "Deep Learning for Computer Vision",
      readmePath: "assignment_1/README.md",
      basePath: "assignment_1",
      links: [
        { label: "Demo Video",    url: "https://youtu.be/9KDi4yDItBk" },
        { label: "Presentation",  url: "https://youtu.be/7qu4bMMTxa0" },
        { label: "Report PDF",    url: "assignment_1/report.pdf" },
        { label: "Landing Page",  url: "https://hhoanggt.github.io/DL-for-CV/" }
      ]
    },
    {
      id: "assignment_2",
      title: "Assignment 2",
      subtitle: "Image Segmentation",
      readmePath: "assignment_2/README.md",
      basePath: "assignment_2",
      links: []
    },
    {
      id: "exercise",
      title: "Exercise",
      subtitle: "CIFAR-10 Deep Learning Portfolio",
      readmePath: "excercise/README.md",
      basePath: "excercise",
      links: []
    }
  ]
};

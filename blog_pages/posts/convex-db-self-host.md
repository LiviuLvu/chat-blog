---
title: "💻 Convex Hosting on a Home Lab with Docker and Orbstack"
date: "2025-11-20"
summary: "Self-hosting modern application backends is often a puzzle of configuration, especially when working with newer architectures and tools. Convex, with its focus on full-stack development, offers a self-hosted option that's perfect for a **home lab** environment."
# description: ""
tags: ["Home lab", "networking", "Convex DB", "Orbstack"]
author: "Liviu Iancu"
weight: 1
series: ["Devops"]
# categories: [""]
# aliases: [""]
# cover:
#   image: images/img.png
#   caption: "Description [text](https://link.somewhere/)"
---
Self-hosting modern application backends is often a puzzle of configuration, especially when working with newer architectures and tools. Convex, with its focus on full-stack development, offers a self-hosted option that's perfect for my **home lab** environment.

This article details the key adjustments I made to successfully run the **Convex DB, dashboard, and a basic chat application frontend** using Docker, specifically testing on an **Orbstack** setup on a **MacOS Mini M4 ARM** machine.

---

## 🛠️ The Self-Hosting Setup

The foundation of this setup follows the official Convex self-hosted instructions but required crucial configuration changes to ensure the containers could correctly communicate across the Docker network, especially in the context of Orbstack's local hostname resolution.

### 1. Frontend Container Configuration

The frontend (FE) application needs to know where to find the self-hosted Convex backend, and it must also listen on an accessible address within the container.

* **Convex Self-Hosted URL:** In your Convex application's `.env.local`, specify the hostname resolvable by the FE container. In an Orbstack environment, this often looks like an internal `.orb.local` address:

    ```bash
    CONVEX_SELF_HOSTED_URL=https://<providedby>.orb.local
    CONVEX_SELF_HOSTED_ADMIN_KEY=your_admin_key
    ```
    ***Insight:** Using a specific hostname instead of `localhost` or `127.0.0.1` ensures that the connection resolves correctly from **inside** the frontend container to the host or the dedicated Convex backend service.*

* **Frontend Server Binding:** For the container's port mapping to work, the development server must listen on all available network interfaces (`0.0.0.0`) inside the container, not just `localhost`.
    * **package.json** (e.g., for a Vite app):
        ```json
        "dev:frontend": "vite --open --host"
        ```
    * **vite.config.mts** (Crucial for host access):
        ```typescript
        export default defineConfig({
          // ...
          server: {
            host: true, // Listen on 0.0.0.0
            allowedHosts: [
              '<frontend-container-url>.orb.local', // Allow host access
            ]
          }
        });
        ```
    The above changes successfully run a basic chat app.

* **Port Mapping:** Expose the frontend port from the container to your host machine.
    ```yaml
    ports:
      - "5173:5173"
    ```

---

### 2. Convex DB Container Configuration

The Convex DB and Convex Dashboard containers are managed via `docker-compose`. Network configuration and environment variables were the primary points of adjustment.

* **Shared Network:** The FE and DB containers must share the same Docker network to communicate seamlessly. I used an already created bridge network named `echoapp`:

    ```yaml
    # compose.yaml
    ...
        networks:
          - convex-net

    networks:
      convex-net:
        external: true
        name: echoapp # Needs to match the FE container's network
        driver: bridge
    ```

* **Origin Environment Variables:** These variables dictate how Convex generates URLs internally for various services. All must point to the resolvable host address.

    * **compose.yaml (Orbstack Links):**
        ```yaml
        - CONVEX_CLOUD_ORIGIN=https://<providedby>.orb.local
        - CONVEX_SITE_ORIGIN=https://<providedby>.orb.local
        ```
    * **.env file for convex docker compose:**
        ```bash
        CONVEX_CLOUD_ORIGIN='https://<providedby>.orb.local'
        CONVEX_SITE_ORIGIN='https://<providedby>.orb.local'
        NEXT_PUBLIC_DEPLOYMENT_URL='https://<providedby>.orb.local'
        ```
    ***Insight:** These `ORIGIN` variables are the key to making the Convex dashboard and application routing work correctly behind a self-hosted proxy/gateway.*

---

### 3. Accessing the Dashboard

Once all containers are up and running with the correct networking and environment variables, you need the administrative key to access the self-hosted dashboard.

* **Generate DB Access Key:** This utility script is inside the database container:
    ```bash
    docker exec <your-database-container-name> ./generate_admin_key.sh
    ```

* **Convex Dashboard UI Login:**
    * **URL:** `https://<providedby>.orb.local`
    * **API Key:** The key generated with `./generate_admin_key.sh`.

This process successfully brought up the full Convex stack, allowing for local development against a fully self-hosted database instance.

---

### 🚀 Conclusion: Own Your Stack

Successfully navigating these networking hurdles demonstrates the power and flexibility of self-hosting. Taking the time to debug the communication between containers and properly define network origins turns a complex setup into a robust, local development environment.

**What's your experience?** Have you self-hosted similar backends in your home lab? Share your experience and any crucial configuration tweaks you discovered!
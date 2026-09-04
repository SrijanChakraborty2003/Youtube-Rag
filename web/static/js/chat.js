document.addEventListener('DOMContentLoaded', () => {
    // Auth elements
    const authModal = document.getElementById('auth-modal');
    const emailForm = document.getElementById('email-form');
    const otpForm = document.getElementById('otp-form');
    const emailInput = document.getElementById('auth-email-input');
    const otpInput = document.getElementById('auth-otp-input');
    const backToEmailBtn = document.getElementById('back-to-email-btn');
    const otpHint = document.getElementById('otp-hint');
    const userEmailDisplay = document.getElementById('user-email-display');
    const logoutBtn = document.getElementById('logout-btn');

    // Chat management elements
    const chatListEl = document.getElementById('chat-list');
    const chatCountBadge = document.getElementById('chat-count-badge');
    const openNewChatBtn = document.getElementById('open-new-chat-btn');
    const welcomeNewChatBtn = document.getElementById('welcome-new-chat-btn');
    const newChatModal = document.getElementById('new-chat-modal');
    const closeNewChatModal = document.getElementById('close-new-chat-modal');
    const cancelNewChatBtn = document.getElementById('cancel-new-chat-btn');
    const newChatForm = document.getElementById('new-chat-form');
    const newChatTitle = document.getElementById('new-chat-title');
    const newChatUrl = document.getElementById('new-chat-url');

    // Add video modal elements
    const openAddVideoBtn = document.getElementById('open-add-video-btn');
    const addVideoModal = document.getElementById('add-video-modal');
    const closeAddVideoModal = document.getElementById('close-add-video-modal');
    const cancelAddVideoBtn = document.getElementById('cancel-add-video-btn');
    const addVideoForm = document.getElementById('add-video-form');
    const addVideoUrl = document.getElementById('add-video-url');
    const deleteCurrentChatBtn = document.getElementById('delete-current-chat-btn');

    // Rename chat elements
    const renameChatBtn = document.getElementById('rename-chat-btn');
    const renameChatModal = document.getElementById('rename-chat-modal');
    const closeRenameChatModal = document.getElementById('close-rename-chat-modal');
    const cancelRenameChatBtn = document.getElementById('cancel-rename-chat-btn');
    const renameChatForm = document.getElementById('rename-chat-form');
    const renameChatTitleInput = document.getElementById('rename-chat-title-input');
    let chatToRenameId = null;

    // Active Chat elements

    const activeChatTitle = document.getElementById('active-chat-title');
    const activeChatVideos = document.getElementById('active-chat-videos');
    const chatMessages = document.getElementById('chat-messages');
    const welcomeCard = document.getElementById('welcome-card');
    const chatForm = document.getElementById('chat-form');
    const queryInput = document.getElementById('user-query-input');
    const sendBtn = document.getElementById('send-btn');
    const toggleSidebarBtn = document.getElementById('toggle-sidebar-btn');
    const headerSidebarToggleBtn = document.getElementById('header-sidebar-toggle-btn');
    const sidebar = document.getElementById('sidebar');
    const toastContainer = document.getElementById('toast-container');
    const otpStatusBanner = document.getElementById('otp-status-banner');

    // State
    let currentUser = null;
    let activeChatId = null;
    let lastCompletedCheck = Date.now() / 1000;
    let pollInterval = null;

    // Helper for API auth header
    function getAuthHeaders() {
        const token = localStorage.getItem('rag_session_token');
        return {
            'Content-Type': 'application/json',
            ...(token ? { 'Authorization': `Bearer ${token}` } : {})
        };
    }

    // =========================================================================
    // 1. AUTHENTICATION FLOW
    // =========================================================================
    async function checkAuth() {
        const token = localStorage.getItem('rag_session_token');
        if (!token) {
            showAuthModal();
            return;
        }

        try {
            const res = await fetch('/api/auth/me', { headers: getAuthHeaders() });
            const data = await res.json();
            if (data.status === 'authenticated') {
                currentUser = data.user;
                onAuthenticated();
            } else {
                showAuthModal();
            }
        } catch (e) {
            showAuthModal();
        }
    }

    function showAuthModal() {
        authModal.classList.remove('hidden');
        emailForm.classList.remove('hidden');
        otpForm.classList.add('hidden');
        if (otpStatusBanner) otpStatusBanner.style.display = 'none';
        emailInput.value = '';
        otpInput.value = '';
        emailInput.focus();
    }

    function hideAuthModal() {
        authModal.classList.add('hidden');
    }

    function onAuthenticated() {
        hideAuthModal();
        userEmailDisplay.textContent = currentUser.email;
        loadChats();
        startBackgroundPolling();
    }

    emailForm.addEventListener('submit', async (e) => {
        e.preventDefault();
        const email = emailInput.value.trim();
        if (!email) return;

        const submitBtn = emailForm.querySelector('button[type="submit"]');
        submitBtn.disabled = true;
        submitBtn.querySelector('span').textContent = 'Sending code...';

        try {
            const res = await fetch('/api/auth/request-otp', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ email })
            });
            const data = await res.json();
            if (data.status === 'success') {
                emailForm.classList.add('hidden');
                otpForm.classList.remove('hidden');
                otpInput.value = '';
                otpInput.focus();

                if (otpStatusBanner) {
                    if (data.smtp_sent) {
                        otpStatusBanner.className = 'status-banner success';
                        otpStatusBanner.innerHTML = `<strong>✅ Verification Code Dispatched!</strong><br>A 6-digit code has been sent to <strong>${escapeHtml(email)}</strong>.<br>Please check your inbox (and Spam folder) and enter it below.`;
                    } else {
                        otpStatusBanner.className = 'status-banner info';
                        otpStatusBanner.innerHTML = `<strong>📧 Verification Code Dispatched</strong><br>A code has been generated for <strong>${escapeHtml(email)}</strong>.<br>Please check your email (or server terminal logs) and manually enter the 6-digit code below.`;
                    }
                }
                otpHint.textContent = 'Enter the 6-digit code received in your email.';
            } else {
                alert(data.message || 'Error sending code.');
            }
        } catch (err) {
            alert('Failed to request verification code: ' + err);
        } finally {
            submitBtn.disabled = false;
            submitBtn.querySelector('span').textContent = 'Send Verification Code';
        }
    });


    backToEmailBtn.addEventListener('click', () => {
        otpForm.classList.add('hidden');
        emailForm.classList.remove('hidden');
        emailInput.focus();
    });

    otpForm.addEventListener('submit', async (e) => {
        e.preventDefault();
        const email = emailInput.value.trim();
        const code = otpInput.value.trim();
        if (!email || !code) return;

        const submitBtn = otpForm.querySelector('button[type="submit"]');
        submitBtn.disabled = true;

        try {
            const res = await fetch('/api/auth/verify-otp', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ email, code })
            });
            const data = await res.json();
            if (data.status === 'success') {
                localStorage.setItem('rag_session_token', data.session_token);
                currentUser = data.user;
                onAuthenticated();
            } else {
                alert(data.message || 'Invalid code.');
            }
        } catch (err) {
            alert('Verification failed.');
        } finally {
            submitBtn.disabled = false;
        }
    });

    logoutBtn.addEventListener('click', async () => {
        if (!confirm('Are you sure you want to log out?')) return;
        try {
            await fetch('/api/auth/logout', { method: 'POST', headers: getAuthHeaders() });
        } catch (e) {}
        localStorage.removeItem('rag_session_token');
        currentUser = null;
        activeChatId = null;
        resetChatWindow();
        showAuthModal();
    });

    // =========================================================================
    // 2. CHAT MANAGEMENT & SWITCHING (REAL-TIME REACTIVE SYNC)
    // =========================================================================
    async function loadChats() {
        try {
            const res = await fetch('/api/chats', { headers: getAuthHeaders() });
            const data = await res.json();
            if (data.status === 'success') {
                const chats = data.chats || [];
                syncChatsState(chats);

                // If no chat is actively selected, restore saved chat or select the first chat
                if (!activeChatId && chats.length > 0) {
                    const savedId = localStorage.getItem('rag_active_chat_id');
                    if (savedId && chats.some(c => c.chat_id === savedId)) {
                        selectChat(savedId);
                    } else {
                        selectChat(chats[0].chat_id);
                    }
                }
            }
        } catch (e) {
            console.warn('Failed to load chats:', e);
        }
    }


    function syncChatsState(chats) {
        chatCountBadge.textContent = `${chats.length} chat${chats.length === 1 ? '' : 's'}`;

        if (chats.length === 0) {
            chatListEl.innerHTML = '<div style="color: var(--text-dim); font-size: 0.8rem; padding: 8px 4px;">No chats yet. Click "New Video Chat" to start.</div>';
            return;
        }

        // Remove placeholder if present
        const placeholder = chatListEl.querySelector('.loading-shimmer');
        if (placeholder) chatListEl.innerHTML = '';

        // Track existing IDs in DOM
        const currentChatIds = new Set(chats.map(c => c.chat_id));
        document.querySelectorAll('.chat-item').forEach(item => {
            const id = item.getAttribute('data-chat-id');
            if (!currentChatIds.has(id)) item.remove();
        });

        chats.forEach(chat => {
            const videoCount = (chat.videos || []).length;
            const chunkCount = chat.total_chunks || 0;
            const activeJob = chat.active_job;

            let item = chatListEl.querySelector(`.chat-item[data-chat-id="${chat.chat_id}"]`);

            if (!item) {
                item = document.createElement('div');
                item.className = `chat-item ${chat.chat_id === activeChatId ? 'active' : ''}`;
                item.setAttribute('data-chat-id', chat.chat_id);
                item.innerHTML = `
                    <div class="chat-item-header">
                        <span class="chat-item-title" title="${escapeHtml(chat.title)}">${escapeHtml(chat.title)}</span>
                        <div class="chat-item-actions">
                            <button class="chat-rename-btn" title="Rename conversation">✏️</button>
                            <button class="chat-delete-btn" title="Delete chat and vector knowledge">&times;</button>
                        </div>
                    </div>
                    <div class="chat-item-meta">
                        <span>🎬 ${videoCount} video${videoCount === 1 ? '' : 's'} (${chunkCount} chunks)</span>
                    </div>
                `;

                // Select chat on click
                item.addEventListener('click', (e) => {
                    if (e.target.closest('.chat-delete-btn') || e.target.closest('.chat-rename-btn')) return;
                    selectChat(chat.chat_id);
                });

                // Rename chat
                item.querySelector('.chat-rename-btn').addEventListener('click', (e) => {
                    e.stopPropagation();
                    openRenameModal(chat.chat_id, chat.title);
                });

                // Delete chat
                item.querySelector('.chat-delete-btn').addEventListener('click', async (e) => {
                    e.stopPropagation();
                    if (!confirm(`Permanently delete "${chat.title}" and erase all indexed video vectors for this chat?`)) return;
                    await deleteChat(chat.chat_id);
                });

                chatListEl.appendChild(item);
            } else {
                // In-place dynamic update without re-rendering or losing focus
                item.classList.toggle('active', chat.chat_id === activeChatId);

                // 1. Update Title dynamically
                const titleEl = item.querySelector('.chat-item-title');
                if (titleEl && titleEl.textContent !== chat.title) {
                    titleEl.textContent = chat.title;
                    titleEl.title = chat.title;
                }

                // 2. Update Videos and Chunks count dynamically!
                const metaEl = item.querySelector('.chat-item-meta');
                const expectedMeta = `🎬 ${videoCount} video${videoCount === 1 ? '' : 's'} (${chunkCount} chunks)`;
                if (metaEl && metaEl.textContent.trim() !== expectedMeta) {
                    metaEl.innerHTML = `<span>${expectedMeta}</span>`;
                }
            }

            // 3. Update Progress Bar dynamically with live video counter
            let progressContainer = item.querySelector('.mini-progress-container');
            if (activeJob && activeJob.status === 'processing') {
                // Parse video counter e.g. "[2/8 vids]" from activeJob.step
                let counterText = '';
                if (activeJob.step) {
                    const match = activeJob.step.match(/\[(\d+\/\d+\s*vids?)\]/i);
                    if (match) {
                        counterText = match[1];
                    }
                }
                if (!counterText && videoCount > 0) {
                    counterText = `${videoCount} vids`;
                }

                const displayText = counterText 
                    ? `${counterText} · ${activeJob.progress || 10}%` 
                    : `${activeJob.progress || 10}%`;

                const stepTooltip = activeJob.step || 'Processing video ingestion...';

                if (!progressContainer) {
                    progressContainer = document.createElement('div');
                    progressContainer.className = 'mini-progress-container';
                    progressContainer.title = stepTooltip;
                    progressContainer.innerHTML = `
                        <div class="mini-progress-header">
                            <span class="mini-progress-step" style="overflow:hidden; text-overflow:ellipsis; white-space:nowrap; max-width:135px;">${counterText || 'Ingesting'}</span>
                            <span class="mini-progress-text">${displayText}</span>
                        </div>
                        <div class="mini-progress-track">
                            <div class="mini-progress-fill" style="width: ${activeJob.progress || 10}%;"></div>
                        </div>
                    `;
                    item.appendChild(progressContainer);
                } else {
                    progressContainer.title = stepTooltip;
                    const fillEl = progressContainer.querySelector('.mini-progress-fill');
                    const textEl = progressContainer.querySelector('.mini-progress-text');
                    const stepEl = progressContainer.querySelector('.mini-progress-step');
                    if (fillEl) fillEl.style.width = `${activeJob.progress || 10}%`;
                    if (textEl) textEl.textContent = displayText;
                    if (stepEl && counterText) stepEl.textContent = counterText;
                }
            } else if (progressContainer) {
                progressContainer.remove();
            }
        });

        // 4. Synchronize Active Chat Header dynamically (videos, pills, and title)
        if (activeChatId) {
            const activeChatObj = chats.find(c => c.chat_id === activeChatId);
            if (activeChatObj) {
                // Ensure action buttons are always displayed for active chat
                if (openAddVideoBtn) openAddVideoBtn.style.display = 'inline-flex';
                if (deleteCurrentChatBtn) deleteCurrentChatBtn.style.display = 'inline-flex';
                if (renameChatBtn) renameChatBtn.style.display = 'inline-flex';

                // Dynamic header title update
                if (activeChatTitle.textContent !== activeChatObj.title && activeChatObj.title !== 'New Video Chat') {
                    activeChatTitle.textContent = activeChatObj.title;
                }

                // Dynamic header video pills update when new playlist videos complete
                const currentPills = activeChatVideos.querySelectorAll('.video-pill');
                const vids = activeChatObj.videos || [];
                if (vids.length !== currentPills.length) {
                    renderChatVideos(vids);

                    // If welcome card was "No Videos in This Chat", update to "Chat is Ready"
                    const welcomeCard = chatMessages.querySelector('.welcome-card');
                    if (welcomeCard && vids.length > 0) {
                        chatMessages.innerHTML = `
                            <div class="welcome-card">
                                <div class="welcome-icon">⚡</div>
                                <h2>Chat is Ready</h2>
                                <p>Ask any question about the indexed videos in this chat. Every claim cites exact seconds!</p>
                            </div>
                        `;
                    }

                    // Enable chat inputs immediately for available videos
                    queryInput.disabled = false;
                    sendBtn.disabled = false;
                    queryInput.placeholder = `Ask anything about "${activeChatObj.title}"...`;
                }
            }
        }

    }


    async function selectChat(chatId) {
        activeChatId = chatId;
        localStorage.setItem('rag_active_chat_id', chatId);
        document.querySelectorAll('.chat-item').forEach(el => {
            el.classList.toggle('active', el.getAttribute('data-chat-id') === chatId);
        });


        try {
            const res = await fetch(`/api/chats/${chatId}`, { headers: getAuthHeaders() });
            const data = await res.json();
            if (data.status === 'success') {
                const chat = data.chat;
                activeChatTitle.textContent = chat.title;
                renderChatVideos(chat.videos || []);

                // Enable inputs and action buttons
                queryInput.disabled = false;
                sendBtn.disabled = false;
                openAddVideoBtn.style.display = 'inline-flex';
                deleteCurrentChatBtn.style.display = 'inline-flex';
                if (renameChatBtn) renameChatBtn.style.display = 'inline-flex';
                queryInput.placeholder = `Ask anything about "${chat.title}"...`;
                queryInput.focus();


                renderMessages(chat.messages || [], chat.videos || []);
            }
        } catch (e) {
            console.error('Failed to select chat:', e);
        }
    }

    function renderChatVideos(videos) {
        activeChatVideos.innerHTML = '';
        if (!videos || videos.length === 0) {
            activeChatVideos.innerHTML = '<span style="font-size: 0.75rem; color: var(--text-dim);">No videos indexed yet</span>';
            return;
        }

        videos.forEach(v => {
            const pill = document.createElement('a');
            pill.className = 'video-pill';
            pill.href = v.video_url;
            pill.target = '_blank';
            pill.title = `${v.video_title} (${v.chunk_count} chunks)`;
            pill.innerHTML = `<span>🎬 ${escapeHtml(v.video_title)}</span> <span>↗</span>`;
            activeChatVideos.appendChild(pill);
        });
    }

    function renderMessages(messages, videos) {
        chatMessages.innerHTML = '';
        if (messages.length === 0) {
            if (videos.length === 0) {
                chatMessages.innerHTML = `
                    <div class="welcome-card">
                        <div class="welcome-icon">📥</div>
                        <h2>No Videos in This Chat</h2>
                        <p>Click <strong>+ Add Video</strong> in the top-right header to ingest a YouTube video or playlist for this chat.</p>
                    </div>
                `;
            } else {
                chatMessages.innerHTML = `
                    <div class="welcome-card">
                        <div class="welcome-icon">⚡</div>
                        <h2>Chat is Ready</h2>
                        <p>Ask any question about the indexed videos in this chat. Every claim cites exact seconds!</p>
                    </div>
                `;
            }
            return;
        }

        messages.forEach(m => {
            if (m.role === 'user') {
                appendUserMessage(m.content);
            } else {
                const meta = m.metadata || {};
                appendAssistantMessage(m.content, meta.expanded_queries || [], meta.chunks || []);
            }
        });
        chatMessages.scrollTop = chatMessages.scrollHeight;
    }

    async function deleteChat(chatId) {
        try {
            const res = await fetch(`/api/chats/${chatId}`, {
                method: 'DELETE',
                headers: getAuthHeaders()
            });
            const data = await res.json();
            if (data.status === 'success') {
                showToast('Chat Deleted', 'Conversation and vector knowledge erased.', '🗑');
                if (activeChatId === chatId) {
                    activeChatId = null;
                    resetChatWindow();
                }
                loadChats();
            }
        } catch (e) {
            alert('Failed to delete chat.');
        }
    }

    function resetChatWindow() {
        activeChatTitle.textContent = 'Select or Create a Chat';
        activeChatVideos.innerHTML = '';
        openAddVideoBtn.style.display = 'none';
        deleteCurrentChatBtn.style.display = 'none';
        if (renameChatBtn) renameChatBtn.style.display = 'none';
        queryInput.disabled = true;
        sendBtn.disabled = true;
        queryInput.value = '';
        queryInput.placeholder = 'Select a chat from the sidebar...';
        chatMessages.innerHTML = `
            <div class="welcome-card" id="welcome-card">
                <div class="welcome-icon">🎬</div>
                <h2>No Chat Selected</h2>
                <p>Select an existing video chat from the sidebar or click <strong>New Video Chat</strong> to ingest any YouTube video or playlist.</p>
            </div>
        `;
    }

    deleteCurrentChatBtn.addEventListener('click', () => {
        if (!activeChatId) return;
        if (confirm('Are you sure you want to delete this active chat and all its video chunks?')) {
            deleteChat(activeChatId);
        }
    });

    // =========================================================================
    // 3. MODALS (NEW CHAT, ADD VIDEO & RENAME CHAT)
    // =========================================================================
    function openNewChat() {
        newChatModal.classList.remove('hidden');
        newChatTitle.value = '';
        newChatUrl.value = '';
        newChatUrl.focus();
    }
    function closeNewChat() { newChatModal.classList.add('hidden'); }

    openNewChatBtn.addEventListener('click', openNewChat);
    if (welcomeNewChatBtn) welcomeNewChatBtn.addEventListener('click', openNewChat);
    closeNewChatModal.addEventListener('click', closeNewChat);
    cancelNewChatBtn.addEventListener('click', closeNewChat);

    newChatForm.addEventListener('submit', async (e) => {
        e.preventDefault();
        const url = newChatUrl.value.trim();
        const title = newChatTitle.value.trim();
        if (!url) return;

        const btn = document.getElementById('confirm-new-chat-btn');
        btn.disabled = true;
        btn.textContent = 'Creating & Queuing Ingestion...';

        try {
            const res = await fetch('/api/chats', {
                method: 'POST',
                headers: getAuthHeaders(),
                body: JSON.stringify({ url, title })
            });
            const data = await res.json();
            if (data.status === 'success') {
                closeNewChat();
                await loadChats();
                selectChat(data.chat.chat_id);
                showToast('Ingestion Queued', `Processing "${title || 'YouTube Video'}" in background...`, '⚡');
            } else {
                alert(data.message || 'Failed to create chat.');
            }
        } catch (err) {
            alert('Failed to create chat.');
        } finally {
            btn.disabled = false;
            btn.textContent = 'Create & Ingest Video';
        }
    });

    // Add video modal
    const addVideoStatusBanner = document.getElementById('add-video-status-banner');

    openAddVideoBtn.addEventListener('click', () => {
        addVideoModal.classList.remove('hidden');
        addVideoUrl.value = '';
        if (addVideoStatusBanner) {
            addVideoStatusBanner.classList.add('hidden');
            addVideoStatusBanner.textContent = '';
        }
        addVideoUrl.focus();
    });
    function closeAddVideo() {
        addVideoModal.classList.add('hidden');
        if (addVideoStatusBanner) {
            addVideoStatusBanner.classList.add('hidden');
            addVideoStatusBanner.textContent = '';
        }
    }
    closeAddVideoModal.addEventListener('click', closeAddVideo);
    cancelAddVideoBtn.addEventListener('click', closeAddVideo);

    addVideoForm.addEventListener('submit', async (e) => {
        e.preventDefault();
        if (!activeChatId) return;
        const url = addVideoUrl.value.trim();
        if (!url) return;

        if (addVideoStatusBanner) {
            addVideoStatusBanner.classList.add('hidden');
            addVideoStatusBanner.textContent = '';
        }

        const btn = document.getElementById('confirm-add-video-btn');
        btn.disabled = true;
        btn.textContent = 'Adding...';

        try {
            const res = await fetch(`/api/chats/${activeChatId}/videos`, {
                method: 'POST',
                headers: getAuthHeaders(),
                body: JSON.stringify({ url })
            });
            const data = await res.json();
            if (data.status === 'success') {
                closeAddVideo();
                showToast('Video Added', 'Ingestion started in background for this chat.', '📥');
                loadChats();
            } else {
                const errMsg = data.message || 'This video is already added in this chat.';
                if (addVideoStatusBanner) {
                    addVideoStatusBanner.textContent = `⚠️ ${errMsg}`;
                    addVideoStatusBanner.classList.remove('hidden');
                } else {
                    alert(errMsg);
                }
                showToast('Already Added', errMsg, '⚠️');
            }
        } catch (err) {
            if (addVideoStatusBanner) {
                addVideoStatusBanner.textContent = '⚠️ Failed to connect to server.';
                addVideoStatusBanner.classList.remove('hidden');
            } else {
                alert('Failed to add video.');
            }
        } finally {
            btn.disabled = false;
            btn.textContent = 'Add Video';
        }
    });

    // Rename chat modal
    function openRenameModal(chatId, currentTitle) {
        chatToRenameId = chatId;
        renameChatTitleInput.value = currentTitle || '';
        renameChatModal.classList.remove('hidden');
        renameChatTitleInput.focus();
        renameChatTitleInput.select();
    }

    function closeRenameModal() {
        renameChatModal.classList.add('hidden');
        chatToRenameId = null;
    }

    if (renameChatBtn) {
        renameChatBtn.addEventListener('click', () => {
            if (!activeChatId) return;
            openRenameModal(activeChatId, activeChatTitle.textContent);
        });
    }

    if (closeRenameChatModal) closeRenameChatModal.addEventListener('click', closeRenameModal);
    if (cancelRenameChatBtn) cancelRenameChatBtn.addEventListener('click', closeRenameModal);

    if (renameChatForm) {
        renameChatForm.addEventListener('submit', async (e) => {
            e.preventDefault();
            const newTitle = renameChatTitleInput.value.trim();
            if (!newTitle || !chatToRenameId) return;

            const btn = document.getElementById('confirm-rename-chat-btn');
            btn.disabled = true;
            btn.textContent = 'Saving...';

            try {
                const res = await fetch(`/api/chats/${chatToRenameId}`, {
                    method: 'PATCH',
                    headers: getAuthHeaders(),
                    body: JSON.stringify({ title: newTitle })
                });
                const data = await res.json();
                if (data.status === 'success') {
                    closeRenameModal();
                    await loadChats();
                    if (activeChatId === chatToRenameId) {
                        activeChatTitle.textContent = newTitle;
                        queryInput.placeholder = `Ask anything about "${newTitle}"...`;
                    }
                    showToast('Chat Renamed', `Conversation renamed to "${newTitle}"`, '✏️');
                } else {
                    alert(data.message || 'Failed to rename chat.');
                }
            } catch (err) {
                alert('Failed to rename chat.');
            } finally {
                btn.disabled = false;
                btn.textContent = 'Save Title';
            }
        });
    }


    // =========================================================================
    // 4. CHAT MESSAGING
    // =========================================================================
    // Auto-scroll tracking: allows user to scroll up freely without getting yanked back down
    let isAutoScrollEnabled = true;

    chatMessages.addEventListener('scroll', () => {
        // If user is within 60px of the bottom, keep auto-scroll active.
        // If user scrolls up, auto-scroll pauses until they scroll back down.
        const distanceFromBottom = chatMessages.scrollHeight - chatMessages.scrollTop - chatMessages.clientHeight;
        isAutoScrollEnabled = distanceFromBottom <= 60;
    });

    queryInput.addEventListener('input', () => {
        queryInput.style.height = 'auto';
        queryInput.style.height = Math.min(queryInput.scrollHeight, 160) + 'px';
    });

    queryInput.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            chatForm.dispatchEvent(new Event('submit'));
        }
    });

    chatForm.addEventListener('submit', async (e) => {
        e.preventDefault();
        if (!activeChatId) return;
        const text = queryInput.value.trim();
        if (!text) return;

        // Clear welcome card if visible
        const welcome = chatMessages.querySelector('.welcome-card');
        if (welcome) welcome.remove();

        appendUserMessage(text);
        queryInput.value = '';
        queryInput.style.height = 'auto';
        sendBtn.disabled = true;

        isAutoScrollEnabled = true;
        const typingEl = appendTypingIndicator();
        chatMessages.scrollTop = chatMessages.scrollHeight;

        try {
            const res = await fetch(`/api/chats/${activeChatId}/messages`, {
                method: 'POST',
                headers: getAuthHeaders(),
                body: JSON.stringify({ query: text })
            });

            if (!res.ok) {
                const errData = await res.json().catch(() => ({}));
                throw new Error(errData.message || `Server responded with status ${res.status}`);
            }

            const reader = res.body.getReader();
            const decoder = new TextDecoder('utf-8');
            let bufferStr = '';
            let accumulatedAnswer = '';
            let expandedQueries = [];
            let chunks = [];
            let assistantRow = null;
            let messageContentEl = null;

            while (true) {
                const { done, value } = await reader.read();
                if (done) break;

                bufferStr += decoder.decode(value, { stream: true });
                const lines = bufferStr.split('\n');
                bufferStr = lines.pop(); // Keep incomplete line for next iteration

                for (const line of lines) {
                    const trimmed = line.trim();
                    if (!trimmed || !trimmed.startsWith('data:')) continue;
                    const jsonStr = trimmed.slice(5).trim();
                    if (!jsonStr) continue;

                    try {
                        const event = JSON.parse(jsonStr);

                        if (event.event === 'status') {
                            // Silent: do not flash distracting micro-messages
                        } else if (event.event === 'metadata') {
                            if (event.expanded_queries) expandedQueries = event.expanded_queries;
                            if (event.chunks) chunks = event.chunks;
                        } else if (event.event === 'token') {
                            if (!assistantRow) {
                                if (typingEl && typingEl.parentNode) typingEl.remove();
                                assistantRow = createAssistantMessagePlaceholder();
                                messageContentEl = assistantRow.querySelector('.message-content');
                            }
                            accumulatedAnswer += event.delta;
                            messageContentEl.innerHTML = formatMarkdownWithCitations(accumulatedAnswer) + '<span class="streaming-cursor"></span>';
                            if (isAutoScrollEnabled) {
                                chatMessages.scrollTop = chatMessages.scrollHeight;
                            }
                        } else if (event.event === 'done') {
                            if (typingEl && typingEl.parentNode) typingEl.remove();
                            if (!assistantRow) {
                                assistantRow = createAssistantMessagePlaceholder();
                                messageContentEl = assistantRow.querySelector('.message-content');
                            }
                            accumulatedAnswer = event.answer || accumulatedAnswer;
                            messageContentEl.innerHTML = formatMarkdownWithCitations(accumulatedAnswer);

                            // Only attach source / query metadata at bottom upon completion
                            const queriesToUse = event.expanded_queries || expandedQueries;
                            const chunksToUse = event.chunks || chunks;
                            if (event.intent !== 'direct_chat') {
                                updateAssistantAccordion(assistantRow, queriesToUse, chunksToUse);
                            }

                            if (isAutoScrollEnabled) {
                                chatMessages.scrollTop = chatMessages.scrollHeight;
                            }
                        } else if (event.event === 'error') {
                            if (typingEl && typingEl.parentNode) typingEl.remove();
                            appendAssistantMessage(`⚠️ **Error:** ${event.message || 'Failed to generate answer.'}`);
                        }
                    } catch (pe) {
                        console.error('SSE JSON parse error:', pe, jsonStr);
                    }
                }
            }
        } catch (err) {
            if (typingEl && typingEl.parentNode) typingEl.remove();
            appendAssistantMessage(`⚠️ **Connection Error:** Could not connect to backend. (${err.message || err})`);
        } finally {
            sendBtn.disabled = false;
            queryInput.focus();
            if (isAutoScrollEnabled) {
                chatMessages.scrollTop = chatMessages.scrollHeight;
            }
        }
    });

    function appendUserMessage(text) {
        const row = document.createElement('div');
        row.className = 'message-row user';
        row.innerHTML = `
            <div class="message-avatar">You</div>
            <div class="message-bubble">${escapeHtml(text)}</div>
        `;
        chatMessages.appendChild(row);
    }

    function appendTypingIndicator() {
        const row = document.createElement('div');
        row.className = 'message-row assistant typing-row';
        row.innerHTML = `
            <div class="message-avatar">🤖</div>
            <div class="message-bubble">
                <div class="typing-dots">
                    <div class="typing-dot"></div>
                    <div class="typing-dot"></div>
                    <div class="typing-dot"></div>
                </div>
            </div>
        `;
        chatMessages.appendChild(row);
        return row;
    }

    function createAssistantMessagePlaceholder() {
        const row = document.createElement('div');
        row.className = 'message-row assistant';
        row.innerHTML = `
            <div class="message-avatar">🤖</div>
            <div class="message-bubble">
                <div class="message-content"></div>
                <div class="subqueries-slot"></div>
                <div class="sources-slot"></div>
            </div>
        `;
        chatMessages.appendChild(row);
        return row;
    }

    function updateAssistantAccordion(assistantRow, expandedQueries, chunks) {
        if (!assistantRow) return;
        if (expandedQueries && expandedQueries.length > 0) {
            const slot = assistantRow.querySelector('.subqueries-slot');
            if (slot && !slot.innerHTML.trim()) {
                const queryItems = expandedQueries.map(q => `<div class="query-tag">🔍 ${escapeHtml(q)}</div>`).join('');
                slot.innerHTML = `
                    <div class="sub-queries-panel" style="margin-top: 14px;">
                        <div class="accordion-header" onclick="this.nextElementSibling.classList.toggle('open')">
                            <span>⚡ Search Queries Used</span>
                            <span>▼</span>
                        </div>
                        <div class="accordion-content">
                            <div class="query-tag-list">${queryItems}</div>
                        </div>
                    </div>
                `;
            }
        }
        if (chunks && chunks.length > 0) {
            const slot = assistantRow.querySelector('.sources-slot');
            if (slot && !slot.innerHTML.trim()) {
                const sourceItems = chunks.map((c, i) => `
                    <div class="source-item">
                        <div class="source-title">[${i+1}] ${escapeHtml(c.video_title || 'Video')}</div>
                        <div class="source-meta">
                            <span>⏱ ${escapeHtml(c.start_timestamp)} ➔ ${escapeHtml(c.end_timestamp)}</span>
                            <span>Re-rank: ${(c.rerank_score || 0).toFixed(3)}</span>
                            <a href="${c.timestamp_url}" target="_blank" class="source-link">Watch YouTube ↗</a>
                        </div>
                    </div>
                `).join('');
                slot.innerHTML = `
                    <div class="sources-panel" style="margin-top: 8px;">
                        <div class="sources-header" onclick="this.nextElementSibling.classList.toggle('open')">
                            <span>📑 Top ${chunks.length} Precision Chunks (FlashRank Scored)</span>
                            <span>▼</span>
                        </div>
                        <div class="sources-content">${sourceItems}</div>
                    </div>
                `;
            }
        }
    }

    function appendAssistantMessage(answerMarkdown, expandedQueries = [], chunks = []) {
        const row = document.createElement('div');
        row.className = 'message-row assistant';

        let subQueriesHtml = '';
        if (expandedQueries && expandedQueries.length > 0) {
            const queryItems = expandedQueries.map(q => `<div class="query-tag">🔍 ${escapeHtml(q)}</div>`).join('');
            subQueriesHtml = `
                <div class="sub-queries-panel">
                    <div class="accordion-header" onclick="this.nextElementSibling.classList.toggle('open')">
                        <span>⚡ 5 Expanded Queries by gpt-oss:120b-cloud</span>
                        <span>▼</span>
                    </div>
                    <div class="accordion-content">
                        <div class="query-tag-list">${queryItems}</div>
                    </div>
                </div>
            `;
        }

        let sourcesHtml = '';
        if (chunks && chunks.length > 0) {
            const sourceItems = chunks.map((c, i) => `
                <div class="source-item">
                    <div class="source-title">[${i+1}] ${escapeHtml(c.video_title || 'Video')}</div>
                    <div class="source-meta">
                        <span>⏱ ${escapeHtml(c.start_timestamp)} ➔ ${escapeHtml(c.end_timestamp)}</span>
                        <span>Re-rank: ${(c.rerank_score || 0).toFixed(3)}</span>
                        <a href="${c.timestamp_url}" target="_blank" class="source-link">Watch YouTube ↗</a>
                    </div>
                </div>
            `).join('');

            sourcesHtml = `
                <div class="sources-panel">
                    <div class="sources-header" onclick="this.nextElementSibling.classList.toggle('open')">
                        <span>📑 Top ${chunks.length} Precision Chunks (FlashRank Scored)</span>
                        <span>▼</span>
                    </div>
                    <div class="sources-content">${sourceItems}</div>
                </div>
            `;
        }

        const formattedBody = formatMarkdownWithCitations(answerMarkdown);

        row.innerHTML = `
            <div class="message-avatar">🤖</div>
            <div class="message-bubble">
                ${subQueriesHtml}
                ${sourcesHtml}
                <div class="message-content">${formattedBody}</div>
            </div>
        `;
        chatMessages.appendChild(row);
    }

    function formatMarkdownWithCitations(md) {
        if (!md) return '';

        // Convert markdown citations [Anchor @ MM:SS](url) or [Anchor](url) to interactive pills
        md = md.replace(/\[([^\]]+)\]\((https?:\/\/[^\s\)]+)\)/g, (match, anchor, url) => {
            return `<a href="${url}" target="_blank" class="citation-pill" title="Jump to timestamp on YouTube">${anchor}</a>`;
        });

        let html = md
            .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
            .replace(/\*(.*?)\*/g, '<em>$1</em>')
            .replace(/```([a-z]*)\n([\s\S]*?)```/g, '<pre><code>$2</code></pre>')
            .replace(/`([^`]+)`/g, '<code>$1</code>')
            .replace(/\|(.+)\|/g, (match) => {
                const cells = match.split('|').filter(c => c.trim() !== '');
                if (cells.some(c => c.includes('---'))) return '';
                const td = cells.map(c => `<td>${c.trim()}</td>`).join('');
                return `<tr>${td}</tr>`;
            })
            .replace(/^\s*-\s+(.+)$/gm, '<li>$1</li>')
            .replace(/^\s*(\d+)\.\s+(.+)$/gm, '<li><strong>$1.</strong> $2</li>')
            .replace(/^### (.*$)/gim, '<h3>$1</h3>')
            .replace(/^## (.*$)/gim, '<h2>$1</h2>')
            .replace(/^# (.*$)/gim, '<h1>$1</h1>')
            .replace(/\n\n+/g, '</p><p>');

        html = '<p>' + html + '</p>';
        if (html.includes('<tr>')) {
            html = html.replace(/(<tr>[\s\S]*?<\/tr>)+/g, '<div style="overflow-x:auto;"><table>$1</table></div>');
        }
        if (html.includes('<li>')) {
            html = html.replace(/(<li>[\s\S]*?<\/li>)+/g, '<ul>$1</ul>');
        }
        return html;
    }

    function escapeHtml(str) {
        if (!str) return '';
        return str
            .replace(/&/g, '&amp;')
            .replace(/</g, '&lt;')
            .replace(/>/g, '&gt;')
            .replace(/"/g, '&quot;')
            .replace(/'/g, '&#039;');
    }

    // =========================================================================
    // 5. TOAST NOTIFICATIONS & BACKGROUND INGESTION POLLER
    // =========================================================================
    function showToast(title, message, icon = '🎉', onAction = null) {
        const toast = document.createElement('div');
        toast.className = 'toast-popup';
        toast.innerHTML = `
            <div class="toast-icon">${icon}</div>
            <div class="toast-body">
                <div class="toast-title">${escapeHtml(title)}</div>
                <div class="toast-message">${escapeHtml(message)}</div>
                ${onAction ? `<button class="toast-action-btn">${onAction.text}</button>` : ''}
            </div>
            <button class="toast-close" title="Close">&times;</button>
        `;

        toast.querySelector('.toast-close').addEventListener('click', () => toast.remove());
        if (onAction) {
            toast.querySelector('.toast-action-btn').addEventListener('click', () => {
                onAction.callback();
                toast.remove();
            });
        }

        toastContainer.appendChild(toast);
        setTimeout(() => {
            if (toast.parentNode) toast.remove();
        }, 12000);
    }

    function startBackgroundPolling() {
        if (pollInterval) clearInterval(pollInterval);

        pollInterval = setInterval(async () => {
            if (!currentUser) return;

            // 1. Dynamic sync of all chats, video counts, chunk counts, progress bars, and header pills
            try {
                await loadChats();
            } catch (e) {}

            // 2. Check for newly completed ingestion jobs for celebratory toast notifications
            try {
                const res = await fetch(`/api/jobs/completed?since=${lastCompletedCheck}`, { headers: getAuthHeaders() });
                const data = await res.json();
                if (data.status === 'success' && data.jobs && data.jobs.length > 0) {
                    lastCompletedCheck = data.server_time || (Date.now() / 1000);
                    data.jobs.forEach(j => {
                        showToast(
                            'Ingestion Complete!',
                            `"${j.chat_title || 'Video'}" is fully indexed! You can now ask questions about it.`,
                            '🎉',
                            {
                                text: 'Go to Chat',
                                callback: () => selectChat(j.chat_id)
                            }
                        );
                    });
                }
            } catch (e) {}
        }, 2000);
    }


    // Sidebar Toggle (Expand and Collapse)
    function toggleSidebar() {
        sidebar.classList.toggle('collapsed');
        const isCollapsed = sidebar.classList.contains('collapsed');
        toggleSidebarBtn.textContent = isCollapsed ? '›' : '‹';
        if (headerSidebarToggleBtn) {
            headerSidebarToggleBtn.title = isCollapsed ? 'Open Sidebar (Show Chats)' : 'Hide Sidebar';
            headerSidebarToggleBtn.classList.toggle('active', isCollapsed);
        }
    }

    if (toggleSidebarBtn) {
        toggleSidebarBtn.addEventListener('click', toggleSidebar);
    }
    if (headerSidebarToggleBtn) {
        headerSidebarToggleBtn.addEventListener('click', toggleSidebar);
    }

    // Start App
    checkAuth();
});

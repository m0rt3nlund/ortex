//! A single, persistent OS thread that owns every onnxruntime `Session` and performs
//! every CUDA-touching operation (init/run/shutdown, across every loaded model).

use ort::session::Session;
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::mpsc::{self, Sender};
use std::sync::OnceLock;
use std::thread;

type Job = Box<dyn FnOnce(&mut HashMap<u64, Session>) + Send + 'static>;

fn sender() -> &'static Sender<Job> {
    static SENDER: OnceLock<Sender<Job>> = OnceLock::new();
    SENDER.get_or_init(|| {
        let (tx, rx) = mpsc::channel::<Job>();
        thread::Builder::new()
            .name("ortex-cuda-worker".into())
            .spawn(move || {
                let mut sessions: HashMap<u64, Session> = HashMap::new();
                for job in rx {
                    job(&mut sessions);
                }
            })
            .expect("failed to spawn ortex CUDA worker thread");
        tx
    })
}

static NEXT_ID: AtomicU64 = AtomicU64::new(0);

/// Allocates a new, unique session id
pub fn next_id() -> u64 {
    NEXT_ID.fetch_add(1, Ordering::Relaxed)
}

/// Runs `f` on the single dedicated CUDA worker thread
pub fn run<F, R>(f: F) -> R
where
    F: FnOnce(&mut HashMap<u64, Session>) -> R + Send + 'static,
    R: Send + 'static,
{
    let (reply_tx, reply_rx) = mpsc::channel();
    let job: Job = Box::new(move |sessions| {
        let _ = reply_tx.send(f(sessions));
    });
    sender()
        .send(job)
        .expect("ortex CUDA worker thread has terminated");
    reply_rx
        .recv()
        .expect("ortex CUDA worker thread dropped the reply channel without responding")
}

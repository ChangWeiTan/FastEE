package multiThreading;

import application.Application;

import java.util.List;
import java.util.concurrent.*;

public class MultiThreadedTask {
    /**
     * Stores some of the multi-threading tasks for PF and TS-CHIEF
     */
    private ExecutorService executor;
    private int poolSize;

    public MultiThreadedTask() {
        // initialize
        if (Application.numThreads == 0) { // auto
            Application.numThreads = Runtime.getRuntime().availableProcessors();
        }

        setExecutor(Executors.newFixedThreadPool(Application.numThreads));

        // this is important for slurm jobs because
        // Runtime.getRuntime().availableProcessors() does not equal SBATCH argument
        // cpus-per-task
        if (executor instanceof ThreadPoolExecutor) {
            this.poolSize = ((ThreadPoolExecutor) executor).getMaximumPoolSize();
        }
    }

    public MultiThreadedTask(int numThreads) {
        numThreads = Math.min(numThreads, Runtime.getRuntime().availableProcessors());
        setExecutor(Executors.newFixedThreadPool(numThreads));

        // this is important for slurm jobs because
        // Runtime.getRuntime().availableProcessors() does not equal SBATCH argument
        // cpus-per-task
        if (executor instanceof ThreadPoolExecutor) {
            this.poolSize = ((ThreadPoolExecutor) executor).getMaximumPoolSize();
        }
    }

    public ExecutorService getExecutor() {
        return executor;
    }

    public void setExecutor(ExecutorService executor) {
        this.executor = executor;
    }

    public ThreadPoolExecutor getThreadPool() {
        if (executor instanceof ThreadPoolExecutor) {
            return ((ThreadPoolExecutor) executor);
        } else {
            return null;
        }
    }

    public static void invokeParallelTasks(List<Callable<Integer>> tasks, MultiThreadedTask parallelTasks) throws Exception {
        List<Future<Integer>> results = parallelTasks.getExecutor().invokeAll(tasks);
        if (Application.verbose > 2) {
            System.out.println("after  -- parallel_tasks.getExecutor().invokeAll(tasks): ");
        }

        for (Future<Integer> futureInt : results) {
            try {
                futureInt.get();
            } catch (ExecutionException ex) {
                ex.getCause().printStackTrace();
                throw new Exception("Error...");
            }
        }
    }
}
